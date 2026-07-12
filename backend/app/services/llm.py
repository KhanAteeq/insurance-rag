# LLM interface service
# ============================================================================
# app/services/llm.py
# ----------------------------------------------------------------------------
# STAGE 7 of the RAG pipeline: the LOCAL LLM (Ollama + llama3.1:8b).
#
# This is the "G" (Generation) in RAG. It takes the chat messages the prompt
# builder produced and asks the local Ollama server to generate an answer.
#
# WHY OLLAMA?
#   * Runs llama3.1:8b entirely on your machine (no cloud, no API key).
#   * Exposes a simple HTTP server on http://localhost:11434.
#   * The official `ollama` Python client wraps it nicely.
#
# WHAT THIS FILE PROVIDES:
#   * A health check (is Ollama up? is the model pulled?).
#   * Blocking generation  -> generate() returns the whole answer.
#   * Streaming generation -> stream() yields tokens as they arrive (for a
#     live "typing" effect in the UI).
#   * Robust timeout + error handling so a slow/unavailable model surfaces a
#     clean, typed error instead of hanging the API.
#
# No LangChain — we call the Ollama client directly.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import httpx
import ollama

from app.config import settings
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Typed errors so the API can distinguish "Ollama is down" from "bad request".
# ----------------------------------------------------------------------------
class LLMError(Exception):
    """Base class for LLM problems."""


class LLMUnavailableError(LLMError):
    """The Ollama server is not reachable."""


class LLMModelMissingError(LLMError):
    """The requested model is not pulled on the Ollama server."""


class LLMTimeoutError(LLMError):
    """Generation took longer than the configured timeout."""


# ----------------------------------------------------------------------------
# Result object for a completed (blocking) generation.
# ----------------------------------------------------------------------------
@dataclass
class LLMResult:
    """The outcome of a blocking generate() call."""
    text: str
    model: str
    took_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ----------------------------------------------------------------------------
# Client construction (singleton).
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    """Return a cached Ollama client bound to the configured host + timeout.

    One client is reused across requests (connection pooling).
    """
    return ollama.Client(host=settings.ollama_host,
                         timeout=settings.ollama_request_timeout)


def _options() -> dict:
    """Generation options passed to Ollama on every call."""
    return {
        "temperature": settings.ollama_temperature,  # low => factual/deterministic
        "num_ctx": settings.ollama_num_ctx,          # context window size
    }


# ----------------------------------------------------------------------------
# Azure OpenAI backend (used when settings.backend_mode == "azure").
# Note: gpt-5 family models only accept the default temperature and require
# `max_completion_tokens` (not `max_tokens`).
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _azure_client():
    """Cached Azure OpenAI client."""
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _azure_ready() -> None:
    """Raise a typed error if Azure OpenAI is not configured."""
    if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
        raise LLMUnavailableError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY (and BACKEND_MODE=azure)."
        )


def _map_azure_error(exc: Exception) -> LLMError:
    """Translate an OpenAI SDK exception into our typed LLM errors."""
    name = type(exc).__name__
    text = str(exc)
    if "Timeout" in name:
        return LLMTimeoutError(f"Azure OpenAI timed out: {text}")
    if "Authentication" in name or "PermissionDenied" in name:
        return LLMUnavailableError(f"Azure OpenAI auth failed: {text}")
    if "Connection" in name:
        return LLMUnavailableError(f"Cannot reach Azure OpenAI: {text}")
    return LLMError(f"Azure OpenAI error: {text}")


def _azure_extra() -> dict:
    """Optional extra params (e.g. reasoning_effort for gpt-5 models)."""
    extra: dict = {}
    if settings.azure_reasoning_effort:
        extra["reasoning_effort"] = settings.azure_reasoning_effort
    return extra


def _generate_azure(messages: list[dict]) -> "LLMResult":
    """Blocking generation via Azure OpenAI chat completions."""
    _azure_ready()
    client = _azure_client()
    try:
        with timer("llm", log) as t:
            resp = client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=messages,
                max_completion_tokens=settings.azure_max_completion_tokens,
                **_azure_extra(),
            )
    except Exception as exc:
        raise _map_azure_error(exc)

    content = resp.choices[0].message.content if resp.choices else ""
    usage = getattr(resp, "usage", None)
    result = LLMResult(
        text=(content or "").strip(),
        model=settings.azure_openai_chat_deployment,
        took_ms=t["ms"],
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
    log.info("Azure LLM produced %d chars in %.0f ms.", len(result.text), t["ms"])
    return result


def _stream_azure(messages: list[dict]) -> Iterator[str]:
    """Streaming generation via Azure OpenAI chat completions."""
    _azure_ready()
    client = _azure_client()
    try:
        stream_iter = client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=messages,
            max_completion_tokens=settings.azure_max_completion_tokens,
            stream=True,
            **_azure_extra(),
        )
        for chunk in stream_iter:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                yield piece
    except Exception as exc:
        raise _map_azure_error(exc)


# ----------------------------------------------------------------------------
# Health check.
# ----------------------------------------------------------------------------
def health() -> dict:
    """Report whether Ollama is reachable and the target model is available.

    Returns a dict like:
        {"reachable": True, "model": "llama3.1:8b", "model_available": True,
         "models": [...]}
    Never raises — designed for the /health endpoint.
    """
    if settings.backend_mode == "azure":
        configured = bool(settings.azure_openai_endpoint and settings.azure_openai_api_key)
        return {
            "reachable": configured,
            "model": settings.azure_openai_chat_deployment,
            "model_available": configured,
            "models": [settings.azure_openai_chat_deployment] if configured else [],
        }

    info: dict = {
        "reachable": False,
        "model": settings.ollama_model,
        "model_available": False,
        "models": [],
    }
    try:
        # A short, independent HTTP probe so a hung model can't block /health.
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        names = [m.get("name", "") for m in data.get("models", [])]
        info["reachable"] = True
        info["models"] = names
        # Model tag may be "llama3.1:8b" or "llama3.1:8b-instruct-q4..." etc.
        info["model_available"] = any(
            n == settings.ollama_model or n.startswith(settings.ollama_model)
            for n in names
        )
    except Exception as exc:
        log.warning("Ollama health check failed: %s", exc)
    return info


def ensure_ready() -> None:
    """Raise a typed error if Ollama or the model is not ready.

    Called before generation so failures are clear and early.
    """
    if settings.backend_mode == "azure":
        _azure_ready()
        return
    status = health()
    if not status["reachable"]:
        raise LLMUnavailableError(
            f"Ollama server not reachable at {settings.ollama_host}. "
            f"Is it running? Try: `ollama serve`."
        )
    if not status["model_available"]:
        raise LLMModelMissingError(
            f"Model '{settings.ollama_model}' is not available on Ollama. "
            f"Pull it with: `ollama pull {settings.ollama_model}`. "
            f"Available: {status['models']}"
        )


# ----------------------------------------------------------------------------
# Blocking generation.
# ----------------------------------------------------------------------------
def generate(messages: list[dict]) -> LLMResult:
    """Generate a complete answer (blocking) from chat messages.

    Args:
        messages: the list[dict] from BuiltPrompt.to_messages().

    Returns:
        LLMResult with the full answer text and timing/usage.

    Raises:
        LLMUnavailableError / LLMModelMissingError / LLMTimeoutError / LLMError
    """
    if settings.backend_mode == "azure":
        return _generate_azure(messages)

    ensure_ready()

    try:
        with timer("llm", log) as t:
            response = _client().chat(
                model=settings.ollama_model,
                messages=messages,
                options=_options(),
                stream=False,
            )
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(
            f"LLM timed out after {settings.ollama_request_timeout}s."
        ) from exc
    except ollama.ResponseError as exc:
        # e.g. model not found at call time
        raise LLMError(f"Ollama responded with an error: {exc}") from exc
    except (httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailableError(f"Lost connection to Ollama: {exc}") from exc
    except Exception as exc:
        raise LLMError(f"LLM generation failed: {exc}") from exc

    text = (response.get("message", {}) or {}).get("content", "") or ""
    text = text.strip()

    result = LLMResult(
        text=text,
        model=settings.ollama_model,
        took_ms=t["ms"],
        prompt_tokens=response.get("prompt_eval_count"),
        completion_tokens=response.get("eval_count"),
    )
    log.info("LLM produced %d chars in %.0f ms (prompt_tokens=%s, completion_tokens=%s).",
             len(text), t["ms"], result.prompt_tokens, result.completion_tokens)
    return result


# ----------------------------------------------------------------------------
# Streaming generation.
# ----------------------------------------------------------------------------
def stream(messages: list[dict]) -> Iterator[str]:
    """Yield the answer token-by-token as Ollama generates it.

    Use this to power a live "typing" effect. Each yielded value is a small
    text fragment; concatenating them reproduces the full answer.

    Raises the same typed errors as generate() (before the first token).
    """
    if settings.backend_mode == "azure":
        yield from _stream_azure(messages)
        return

    ensure_ready()

    try:
        start = timer("llm_stream", log)
        with start:
            stream_iter = _client().chat(
                model=settings.ollama_model,
                messages=messages,
                options=_options(),
                stream=True,
            )
            for part in stream_iter:
                piece = (part.get("message", {}) or {}).get("content", "")
                if piece:
                    yield piece
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(
            f"LLM stream timed out after {settings.ollama_request_timeout}s."
        ) from exc
    except ollama.ResponseError as exc:
        raise LLMError(f"Ollama responded with an error: {exc}") from exc
    except (httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailableError(f"Lost connection to Ollama: {exc}") from exc
    except Exception as exc:
        raise LLMError(f"LLM streaming failed: {exc}") from exc


def warm_up() -> None:
    """Send a tiny prompt so the model is loaded into memory before real use.

    Ollama loads model weights on first use; this hides that latency from the
    first real user. Best-effort — never raises.
    """
    if settings.backend_mode == "azure":
        return  # remote model; nothing to warm, and avoids spending tokens
    try:
        ensure_ready()
        _client().chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            options={"temperature": 0.0, "num_ctx": 512},
            stream=False,
        )
        log.info("LLM warm-up complete for model '%s'.", settings.ollama_model)
    except Exception as exc:
        log.warning("LLM warm-up skipped: %s", exc)
