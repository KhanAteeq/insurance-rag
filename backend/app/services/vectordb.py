# Vector database service
# ============================================================================
# app/services/vectordb.py
# ----------------------------------------------------------------------------
# STAGE 4 of the RAG pipeline: STORE and SEARCH vectors with ChromaDB.
#
# WHAT IS A VECTOR DATABASE?
#   A normal database finds rows by exact key/value ("WHERE id = 5").
#   A VECTOR database finds rows by *nearness in meaning*: give it a query
#   vector and it returns the stored vectors closest to it (nearest neighbours),
#   using cosine distance. That's the engine behind semantic retrieval.
#
# WHY CHROMADB?
#   * Runs 100% locally, persists to a folder on disk (survives restarts).
#   * Simple Python API. Zero servers to run.
#   * Uses an HNSW index under the hood for fast approximate nearest-neighbour.
#
# WHAT WE STORE per chunk:
#   id        -> stable unique string (from make_chunk_id)
#   embedding -> the 384-D vector
#   document  -> the chunk TEXT (Chroma calls the raw text the "document")
#   metadata  -> page_number, chunk_number, section, document_name, char_count
#
# OPERATIONS implemented (the spec's CRUD + search + count):
#   upsert (insert/update), delete, get, search, count, reset.
#
# This is a thin, well-typed wrapper (a Repository) around Chroma so the rest
# of the app never touches Chroma's raw client directly.
# ============================================================================

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.chunker import Chunk
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised on any ChromaDB failure (connect, add, query, delete)."""


# ----------------------------------------------------------------------------
# Result type returned by search — a clean object, not Chroma's raw nested dict.
# ----------------------------------------------------------------------------
@dataclass
class SearchResult:
    """One retrieved chunk plus its similarity to the query."""
    id: str
    text: str
    metadata: dict
    distance: float           # cosine DISTANCE (0 = identical, 2 = opposite)

    @property
    def score(self) -> float:
        """Convert cosine distance -> similarity in [0, 1] (higher = better).

        Chroma's cosine distance = 1 - cosine_similarity, so:
            similarity = 1 - distance
        We clamp to [0, 1] for a friendly, display-ready score.
        """
        sim = 1.0 - self.distance
        return max(0.0, min(1.0, sim))


# ----------------------------------------------------------------------------
# The repository wrapper around ChromaDB.
# ----------------------------------------------------------------------------
class VectorStore:
    """A thin, typed Repository over a single ChromaDB collection."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._lock = threading.Lock()

    # --- connection / lifecycle -------------------------------------------
    def _ensure(self) -> chromadb.Collection:
        """Lazily create the persistent client + collection (once)."""
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:  # double-checked locking
                return self._collection
            try:
                settings.ensure_directories()
                # PersistentClient writes to disk so data survives restarts.
                self._client = chromadb.PersistentClient(
                    path=str(settings.chroma_dir),
                    settings=ChromaSettings(anonymized_telemetry=False,
                                            allow_reset=True),
                )
                # get_or_create so re-runs REUSE the existing collection
                # (performance: avoid rebuilding) instead of erroring.
                self._collection = self._client.get_or_create_collection(
                    name=settings.active_collection_name,
                    # cosine space -> matches our normalised embeddings.
                    metadata={"hnsw:space": "cosine"},
                )
                log.info(
                    "ChromaDB ready: collection='%s' at '%s' (count=%d).",
                    settings.collection_name, settings.chroma_dir,
                    self._collection.count(),
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to initialise ChromaDB at '{settings.chroma_dir}': {exc}"
                ) from exc
            return self._collection

    # --- CREATE / UPDATE (upsert) -----------------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk],
                      embeddings: Sequence[Sequence[float]]) -> int:
        """Insert or update chunks + their vectors.

        upsert => if an id already exists it is OVERWRITTEN, otherwise added.
        This is what makes re-ingesting the same PDF idempotent (no dupes).

        Returns the number of chunks written.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"count mismatch."
            )
        if not chunks:
            return 0

        collection = self._ensure()
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata() for c in chunks]
        vectors = [list(v) for v in embeddings]

        try:
            with timer(f"chroma_upsert:{len(ids)}", log):
                collection.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=documents,
                    metadatas=metadatas,
                )
        except Exception as exc:
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

        log.info("Upserted %d chunks into '%s'.", len(ids), settings.collection_name)
        return len(ids)

    # --- READ / SEARCH -----------------------------------------------------
    def search(self, query_embedding: Sequence[float],
               top_k: int | None = None,
               where: dict | None = None) -> list[SearchResult]:
        """Return the top_k nearest chunks to a query vector.

        Args:
            query_embedding: the embedded question (384-D).
            top_k:           how many neighbours to return (default config).
            where:           optional metadata filter, e.g.
                             {"document_name": "surety-claims.pdf"}.
        """
        collection = self._ensure()
        k = top_k or settings.top_k

        # Never ask for more results than exist (Chroma warns / errors otherwise).
        count = collection.count()
        if count == 0:
            log.warning("Search on an EMPTY collection — returning no results.")
            return []
        k = min(k, count)

        try:
            with timer(f"chroma_query:k={k}", log):
                raw = collection.query(
                    query_embeddings=[list(query_embedding)],
                    n_results=k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        # Chroma returns parallel lists nested one level (per query). We sent
        # ONE query, so index [0] on each list.
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for i in range(len(ids)):
            results.append(
                SearchResult(
                    id=ids[i],
                    text=docs[i] if docs else "",
                    metadata=metas[i] if metas else {},
                    distance=float(dists[i]) if dists else 1.0,
                )
            )
        return results

    def get_by_id(self, chunk_id: str) -> SearchResult | None:
        """Fetch a single stored chunk by its id (no vector search)."""
        collection = self._ensure()
        try:
            raw = collection.get(ids=[chunk_id],
                                 include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Chroma get failed: {exc}") from exc

        ids = raw.get("ids", [])
        if not ids:
            return None
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])
        return SearchResult(
            id=ids[0],
            text=docs[0] if docs else "",
            metadata=metas[0] if metas else {},
            distance=0.0,
        )

    # --- DELETE ------------------------------------------------------------
    def delete_by_ids(self, ids: Sequence[str]) -> int:
        """Delete specific chunks by id. Returns how many ids were requested."""
        if not ids:
            return 0
        collection = self._ensure()
        try:
            collection.delete(ids=list(ids))
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc
        log.info("Deleted %d chunks by id.", len(ids))
        return len(ids)

    def delete_document(self, document_name: str) -> None:
        """Delete every chunk belonging to one source document (by metadata)."""
        collection = self._ensure()
        try:
            collection.delete(where={"document_name": document_name})
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete-document failed: {exc}") from exc
        log.info("Deleted all chunks for document '%s'.", document_name)

    # --- COUNT / STATS -----------------------------------------------------
    def count(self) -> int:
        """Total number of chunks stored in the collection."""
        return self._ensure().count()

    def list_documents(self) -> list[dict]:
        """Return per-document stats: name + how many chunks it contributed.

        Reads all metadatas (fine for a local, modest corpus) and aggregates.
        """
        collection = self._ensure()
        total = collection.count()
        if total == 0:
            return []
        try:
            raw = collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Chroma list failed: {exc}") from exc

        counts: dict[str, int] = {}
        for meta in raw.get("metadatas", []) or []:
            name = str(meta.get("document_name", "unknown"))
            counts[name] = counts.get(name, 0) + 1

        return [{"document_name": name, "chunk_count": n}
                for name, n in sorted(counts.items())]

    def peek_chunks(self, limit: int = 10) -> list[dict]:
        """Return a small sample of stored chunks (for the /chunks endpoint)."""
        collection = self._ensure()
        try:
            raw = collection.get(include=["documents", "metadatas"], limit=limit)
        except Exception as exc:
            raise VectorStoreError(f"Chroma peek failed: {exc}") from exc

        ids = raw.get("ids", []) or []
        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []
        out: list[dict] = []
        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "text": (docs[i][:200] + "…") if docs and len(docs[i]) > 200
                else (docs[i] if docs else ""),
                "metadata": metas[i] if metas else {},
            })
        return out

    # --- RESET (danger) ----------------------------------------------------
    def reset(self) -> None:
        """Delete the whole collection and recreate it empty.

        Used by tests / a manual 'reindex from scratch'. Destructive.
        """
        client = self._ensure() and self._client
        if client is None:
            return
        try:
            client.delete_collection(settings.active_collection_name)
        except Exception:
            pass  # collection may not exist yet; ignore.
        self._collection = None
        self._ensure()
        log.warning("Vector store RESET — collection '%s' recreated empty.",
                    settings.collection_name)


# ----------------------------------------------------------------------------
# Singleton accessor — one shared VectorStore for the whole app.
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the process-wide VectorStore singleton."""
    return VectorStore()
