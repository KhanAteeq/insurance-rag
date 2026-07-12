# Embedding generation service
# ============================================================================
# app/services/embedding.py
# ----------------------------------------------------------------------------
# STAGE 3 of the RAG pipeline: turn text into vectors ("embeddings").
#
# WHAT IS AN EMBEDDING?
#   A neural model reads a piece of text and outputs a fixed-length list of
#   floating-point numbers — a "vector" — that captures the text's *meaning*.
#   Texts with similar meaning land close together in this vector space, even
#   if they share no words ("car accident" ~ "vehicle collision").
#
# MODEL: sentence-transformers/all-MiniLM-L6-v2
#   * Runs 100% locally (no API).
#   * Output dimension = 384  (each text -> 384 floats).
#   * Fast, small (~90 MB), great quality-for-size. Ideal for a laptop.
#
# WHY 384 DIMENSIONS?
#   The model was trained to compress meaning into a 384-D space. More
#   dimensions can capture more nuance but cost more storage + slower search.
#   384 is the model's fixed output size — we don't choose it, the model does.
#
# COSINE SIMILARITY (how we compare vectors later):
#   similarity = (A · B) / (|A| * |B|)   -> ranges from -1 (opposite) to
#   +1 (identical direction). If vectors are L2-NORMALISED (length 1), the
#   dot product IS the cosine similarity. We normalise here so downstream
#   search is a cheap dot product.
#
# PERFORMANCE:
#   * The model is loaded LAZILY and cached (singleton) — first call is slow
#     (loads weights), every call after is fast.
#   * We embed in BATCHES for throughput.
# ============================================================================

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Sequence

import numpy as np

from app.config import settings
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding model fails to load or encode text."""


# A lock so that, under concurrent FastAPI requests, we only load the heavy
# model ONCE even if two requests race on the very first call.
_model_lock = threading.Lock()


# ----------------------------------------------------------------------------
# Azure OpenAI embedding backend (used when settings.backend_mode == "azure").
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _azure_embed_client():
    """Cached Azure OpenAI client for embeddings."""
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _embed_texts_azure(texts: list[str]) -> list[list[float]]:
    """Embed texts via the Azure OpenAI embeddings endpoint (1536-d)."""
    client = _azure_embed_client()
    out: list[list[float]] = []
    batch = max(1, settings.embedding_batch_size)
    try:
        with timer(f"azure_embed:{len(texts)}_texts", log):
            for i in range(0, len(texts), batch):
                resp = client.embeddings.create(
                    model=settings.azure_openai_embedding_deployment,
                    input=texts[i:i + batch],
                )
                out.extend([d.embedding for d in resp.data])
    except Exception as exc:
        raise EmbeddingError(f"Azure embedding failed: {exc}") from exc
    return out


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the sentence-transformers model once and cache it (singleton).

    First call downloads the model (only the very first time on a machine)
    and loads weights into memory — this can take several seconds. Every
    subsequent call returns the already-loaded instance instantly.
    """
    with _model_lock:
        from sentence_transformers import SentenceTransformer  # lazy: heavy dep

        model_name = settings.embedding_model_name
        log.info("Loading embedding model '%s' (first call is slow)...", model_name)
        try:
            with timer(f"load_model:{model_name}", log):
                model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

        # Sanity-check the dimension matches our config / vector store schema.
        dim = model.get_sentence_embedding_dimension()
        if dim != settings.embedding_dimension:
            log.warning(
                "Model dimension (%d) != configured embedding_dimension (%d). "
                "Using the model's actual dimension.",
                dim, settings.embedding_dimension,
            )
        log.info("Embedding model ready. Dimension = %d.", dim)
        return model


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the active backend (384 local / 1536 Azure)."""
    if settings.backend_mode == "azure":
        return settings.azure_embedding_dimension
    return _load_model().get_sentence_embedding_dimension()


def warm_up() -> None:
    """Eagerly load the model at startup so the first user request is fast.

    Called from main.py's startup event. Optional but improves UX.
    """
    if settings.backend_mode == "azure":
        return  # embeddings are remote; nothing to preload
    _load_model()


# ----------------------------------------------------------------------------
# Core embedding functions.
# ----------------------------------------------------------------------------
def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of texts into L2-normalised 384-D vectors.

    Args:
        texts: the chunk texts (or any strings) to embed.

    Returns:
        A list of vectors (list[float]); one per input text, same order.

    Raises:
        EmbeddingError: on empty input or model failure.
    """
    if texts is None:
        raise EmbeddingError("embed_texts received None.")

    # Filter guard: sentence-transformers dislikes None entries.
    cleaned = [t if isinstance(t, str) else "" for t in texts]
    if len(cleaned) == 0:
        return []

    # Azure backend: call the remote embeddings endpoint instead of the local model.
    if settings.backend_mode == "azure":
        return _embed_texts_azure(cleaned)

    model = _load_model()

    try:
        with timer(f"embed:{len(cleaned)}_texts", log):
            # normalize_embeddings=True -> every vector has length 1, so a
            # dot product later equals cosine similarity (fast + correct).
            vectors: np.ndarray = model.encode(
                cleaned,
                batch_size=settings.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc

    # Convert numpy -> plain Python lists so ChromaDB / JSON can consume them.
    return vectors.astype(np.float32).tolist()


def embed_text(text: str) -> list[float]:
    """Embed a SINGLE text (e.g. the user's question) into one 384-D vector.

    Convenience wrapper around embed_texts for the retrieval path.
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text.")
    return embed_texts([text])[0]


# ----------------------------------------------------------------------------
# Cosine similarity — provided for completeness / teaching / tests.
# (ChromaDB computes this internally during search, but understanding it and
#  being able to compute it directly is important.)
# ----------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]:
        +1.0 -> identical direction (very similar meaning)
         0.0 -> orthogonal (unrelated)
        -1.0 -> opposite direction

    Formula:  (A · B) / (|A| * |B|)
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))
