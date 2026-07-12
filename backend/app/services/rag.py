# ============================================================================
# app/services/rag.py
# ----------------------------------------------------------------------------
# THE ORCHESTRATOR — the "read path" that ties the whole RAG pipeline together:
#
#     question
#        │  sanitize_question            (utils.helpers  — security)
#        ▼
#     Retriever.retrieve()               (stage 5 — semantic search)
#        │  RetrievalResult (top-k chunks + scores)
#        ▼
#     build_prompt()                     (stage 6 — grounded prompt)
#        │  BuiltPrompt (system + user messages)
#        ▼
#     llm.generate() / llm.stream()      (stage 7 — local Ollama answer)
#        │
#        ▼
#     RagAnswer  (answer text + cited sources + full timing breakdown)
#
# This module is deliberately thin: every stage already lives in its own,
# well-tested service. rag.py only sequences them, measures timing, and shapes
# a single clean result object for the API layer.
#
# SHORT-CIRCUIT: if retrieval finds nothing relevant, we DO NOT call the LLM.
# We return the canonical "I don't know" answer instantly — faster and it makes
# hallucination on an empty knowledge base impossible.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from app.config import settings
from app.services import llm
from app.services.prompt_builder import build_no_context_answer, build_prompt
from app.services.retriever import RetrievalResult, get_retriever
from app.utils.helpers import sanitize_question
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


class RagError(Exception):
    """Raised when the RAG pipeline fails end-to-end."""


# ----------------------------------------------------------------------------
# Timing breakdown (the spec asks us to surface retrieval/LLM latency).
# ----------------------------------------------------------------------------
@dataclass
class RagTimings:
    """Per-stage latency in milliseconds."""
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "retrieval_ms": round(self.retrieval_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


# ----------------------------------------------------------------------------
# The final, API-ready answer object.
# ----------------------------------------------------------------------------
@dataclass
class RagAnswer:
    """Everything the API needs to return for one question."""
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    timings: RagTimings = field(default_factory=RagTimings)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    grounded: bool = True          # False => answered without any context

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "model": self.model,
            "timings": self.timings.as_dict(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "grounded": self.grounded,
        }


# ----------------------------------------------------------------------------
# The service.
# ----------------------------------------------------------------------------
class RagService:
    """High-level Retrieval-Augmented Generation service."""

    def __init__(self, top_k: int | None = None, min_score: float = 0.0) -> None:
        """
        Args:
            top_k:     how many chunks to retrieve per question (config default).
            min_score: similarity floor; chunks below it are discarded.
        """
        self._top_k = top_k or settings.top_k
        self._min_score = min_score
        self._retriever = get_retriever()

    # --- internal: retrieve + build prompt --------------------------------
    def _retrieve(self, question: str, top_k: int | None,
                  document_name: str | None) -> RetrievalResult:
        """Run stage 5 (retrieval) with error wrapping."""
        try:
            return self._retriever.retrieve(
                question=question,
                top_k=top_k or self._top_k,
                document_name=document_name,
                min_score=self._min_score,
            )
        except Exception as exc:
            raise RagError(f"Retrieval stage failed: {exc}") from exc

    # --- public: blocking answer ------------------------------------------
    def answer(self, question: str, *,
               top_k: int | None = None,
               document_name: str | None = None) -> RagAnswer:
        """Answer a question end-to-end (blocking).

        Args:
            question:      the raw user question (will be sanitised here).
            top_k:         override the number of chunks to retrieve.
            document_name: restrict retrieval to a single source document.

        Returns:
            A fully-populated RagAnswer.

        Raises:
            ValueError: if the question is empty/too long (from sanitisation).
            RagError:   if any pipeline stage fails.
        """
        clean_q = sanitize_question(question)  # may raise ValueError -> HTTP 400
        timings = RagTimings()

        with timer("rag_total", log) as total:
            # --- Stage 5: retrieve --------------------------------------
            retrieval = self._retrieve(clean_q, top_k, document_name)
            timings.retrieval_ms = retrieval.took_ms

            # --- Short-circuit: nothing relevant found ------------------
            if retrieval.is_empty:
                log.info("No relevant chunks for %r — returning canonical refusal.",
                         clean_q[:80])
                timings.total_ms = total["ms"]
                return RagAnswer(
                    question=clean_q,
                    answer=build_no_context_answer(),
                    sources=[],
                    model=settings.ollama_model,
                    timings=timings,
                    grounded=False,
                )

            # --- Stage 6: build the grounded prompt ---------------------
            prompt = build_prompt(clean_q, retrieval)

            # --- Stage 7: generate with the local LLM -------------------
            try:
                result = llm.generate(prompt.to_messages())
            except llm.LLMError:
                raise                       # let the API map typed LLM errors
            except Exception as exc:
                raise RagError(f"Generation stage failed: {exc}") from exc

            timings.llm_ms = result.took_ms

        timings.total_ms = total["ms"]

        answer = RagAnswer(
            question=clean_q,
            answer=result.text or build_no_context_answer(),
            sources=retrieval.sources(),
            model=result.model,
            timings=timings,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            grounded=True,
        )
        log.info("Answered %r in %.0f ms (retrieval=%.0f ms, llm=%.0f ms, %d sources).",
                 clean_q[:80], timings.total_ms, timings.retrieval_ms,
                 timings.llm_ms, len(answer.sources))
        return answer

    # --- public: streaming answer -----------------------------------------
    def answer_stream(self, question: str, *,
                      top_k: int | None = None,
                      document_name: str | None = None) -> Iterator[dict]:
        """Answer a question, streaming tokens as they are generated.

        Yields a sequence of event dicts (ready to serialise as SSE / NDJSON):
            {"type": "sources", "sources": [...]}          (once, first)
            {"type": "token",   "token": "..."}            (many)
            {"type": "done",    "timings": {...}, "grounded": true}
            {"type": "error",   "message": "..."}          (on failure)

        This powers the live "typing" effect in the Angular UI.
        """
        try:
            clean_q = sanitize_question(question)
        except ValueError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        timings = RagTimings()
        try:
            with timer("rag_stream_total", log) as total:
                retrieval = self._retrieve(clean_q, top_k, document_name)
                timings.retrieval_ms = retrieval.took_ms

                # Emit the sources first so the UI can render citations early.
                yield {"type": "sources", "sources": retrieval.sources(),
                       "grounded": not retrieval.is_empty}

                if retrieval.is_empty:
                    yield {"type": "token", "token": build_no_context_answer()}
                    timings.total_ms = total["ms"]
                    yield {"type": "done", "timings": timings.as_dict(),
                           "grounded": False, "model": settings.ollama_model}
                    return

                prompt = build_prompt(clean_q, retrieval)

                produced_any = False
                for piece in llm.stream(prompt.to_messages()):
                    produced_any = True
                    yield {"type": "token", "token": piece}

                if not produced_any:
                    yield {"type": "token", "token": build_no_context_answer()}

            timings.total_ms = total["ms"]
            yield {"type": "done", "timings": timings.as_dict(),
                   "grounded": True, "model": settings.ollama_model}

        except llm.LLMError as exc:
            log.error("LLM error during stream: %s", exc)
            yield {"type": "error", "message": str(exc)}
        except Exception as exc:
            log.exception("Unexpected error during RAG stream.")
            yield {"type": "error", "message": f"RAG failed: {exc}"}


# ----------------------------------------------------------------------------
# Singleton accessor — one shared RagService for the whole app.
# ----------------------------------------------------------------------------
_default_service: RagService | None = None


def get_rag_service() -> RagService:
    """Return the process-wide RagService singleton."""
    global _default_service
    if _default_service is None:
        _default_service = RagService()
    return _default_service
