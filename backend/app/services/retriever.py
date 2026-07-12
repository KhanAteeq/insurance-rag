# Document retriever service
# ============================================================================
# app/services/retriever.py
# ----------------------------------------------------------------------------
# STAGE 5 of the RAG pipeline: RETRIEVAL (the "R" in RAG).
#
# Given a natural-language QUESTION, find the few chunks most likely to contain
# the answer:
#
#     question ──▶ embed_text ──▶ vector_store.search(top_k) ──▶ rank/filter
#                                                                    │
#                                                                    ▼
#                                              top-N RetrievedChunk (text +
#                                              metadata + similarity score)
#
# This is the "read path" counterpart to ingest.py. It is deliberately small
# and pure: it does NOT talk to the LLM. It only finds relevant context.
#
# EXTRAS beyond a raw vector search:
#   * A minimum-similarity THRESHOLD so obviously-irrelevant chunks are dropped
#     (helps the LLM say "I don't know" instead of hallucinating).
#   * Optional metadata filtering (e.g. restrict to one document).
#   * A neat, LLM-ready context string builder with source citations.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.services.embedding import embed_text
from app.services.vectordb import SearchResult, get_vector_store
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval fails (embedding or vector search error)."""


# ----------------------------------------------------------------------------
# The object we hand to the prompt builder / API.
# ----------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    """A single relevant chunk with everything the answer layer needs."""
    id: str
    text: str
    score: float                 # similarity in [0, 1] (higher = better)
    document_name: str
    page_number: int
    chunk_number: int
    section: str

    @classmethod
    def from_search(cls, r: SearchResult) -> "RetrievedChunk":
        """Adapt a low-level SearchResult into a rich RetrievedChunk."""
        meta = r.metadata or {}
        return cls(
            id=r.id,
            text=r.text,
            score=r.score,
            document_name=str(meta.get("document_name", "unknown")),
            page_number=int(meta.get("page_number", 0)),
            chunk_number=int(meta.get("chunk_number", 0)),
            section=str(meta.get("section", "")),
        )

    def citation(self) -> str:
        """Human-readable source label, e.g. '[surety-claims.pdf p.6]'."""
        return f"[{self.document_name} p.{self.page_number}]"


@dataclass
class RetrievalResult:
    """The full outcome of a retrieval call."""
    question: str
    chunks: list[RetrievedChunk]
    took_ms: float

    @property
    def is_empty(self) -> bool:
        """True when nothing relevant was found (LLM should say 'I don't know')."""
        return len(self.chunks) == 0

    def context_text(self, max_chars: int | None = None) -> str:
        """Build the CONTEXT block fed to the LLM.

        Each chunk is prefixed with a numbered source citation so the model
        (and the user) can trace every fact back to a page.

        Optionally truncated to `max_chars` to respect the prompt-size limit.
        """
        blocks: list[str] = []
        for i, c in enumerate(self.chunks, start=1):
            blocks.append(f"[Source {i} | {c.citation()} | section: {c.section}]\n{c.text}")
        context = "\n\n".join(blocks)

        if max_chars is not None and len(context) > max_chars:
            context = context[:max_chars].rsplit("\n", 1)[0] + "\n…[truncated]"
        return context

    def sources(self) -> list[dict]:
        """Machine-readable list of sources for the API response."""
        return [
            {
                "id": c.id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "chunk_number": c.chunk_number,
                "section": c.section,
                "score": round(c.score, 4),
            }
            for c in self.chunks
        ]


# ----------------------------------------------------------------------------
# The retriever.
# ----------------------------------------------------------------------------
class Retriever:
    """Turns a question into the most relevant chunks (semantic search)."""

    def __init__(self, min_score: float = 0.0) -> None:
        """
        Args:
            min_score: drop chunks whose similarity is below this threshold.
                       0.0 keeps everything; ~0.2-0.3 filters weak matches.
        """
        self._min_score = min_score

    def retrieve(self, question: str,
                 top_k: int | None = None,
                 document_name: str | None = None,
                 min_score: float | None = None) -> RetrievalResult:
        """Find the top_k most relevant chunks for `question`.

        Args:
            question:      the (already sanitised) user question.
            top_k:         number of chunks to return (default settings.top_k).
            document_name: restrict search to one document via metadata filter.
            min_score:     override the instance similarity threshold.

        Raises:
            RetrievalError: on embedding or vector-search failure.
        """
        if not question or not question.strip():
            raise RetrievalError("Cannot retrieve for an empty question.")

        k = top_k or settings.top_k
        threshold = self._min_score if min_score is None else min_score
        where = {"document_name": document_name} if document_name else None

        with timer("retrieval", log) as t:
            # --- 1. Embed the question ------------------------------------
            try:
                query_vector = embed_text(question)
            except Exception as exc:
                raise RetrievalError(f"Failed to embed question: {exc}") from exc

            # --- 2. Vector search -----------------------------------------
            try:
                raw_results = get_vector_store().search(
                    query_embedding=query_vector,
                    top_k=k,
                    where=where,
                )
            except Exception as exc:
                raise RetrievalError(f"Vector search failed: {exc}") from exc

        # --- 3. Adapt + threshold-filter + sort ---------------------------
        chunks = [RetrievedChunk.from_search(r) for r in raw_results]
        if threshold > 0.0:
            kept = [c for c in chunks if c.score >= threshold]
            dropped = len(chunks) - len(kept)
            if dropped:
                log.info("Retrieval dropped %d/%d chunks below score %.2f.",
                         dropped, len(chunks), threshold)
            chunks = kept

        # Chroma already returns nearest-first, but sort defensively.
        chunks.sort(key=lambda c: c.score, reverse=True)

        result = RetrievalResult(
            question=question,
            chunks=chunks,
            took_ms=t["ms"],
        )

        log.info("Retrieved %d chunk(s) for question: %r (%.1f ms).",
                 len(chunks), question[:80], t["ms"])
        return result


# ----------------------------------------------------------------------------
# Singleton accessor.
# ----------------------------------------------------------------------------
_default_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Return a shared Retriever instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever(min_score=0.0)
    return _default_retriever
