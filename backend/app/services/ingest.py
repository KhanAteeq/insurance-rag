# ============================================================================
# app/services/ingest.py
# ----------------------------------------------------------------------------
# The INGESTION ORCHESTRATOR — wires the first four stages into one pipeline:
#
#     read_pdf  ->  chunk_document  ->  embed_texts  ->  vector_store.upsert
#     (stage 1)      (stage 2)          (stage 3)         (stage 4)
#
# It is the "write path" of RAG: everything needed to take a PDF from disk (or
# an upload) and make it searchable.
#
# KEY FEATURES (from the spec):
#   * "Everything should happen automatically."  -> one call does it all.
#   * "Avoid duplicate ingestion."               -> we hash the file and skip
#     re-processing an identical PDF (unless force=True).
#   * A small on-disk MANIFEST (JSON) records which file hashes we've ingested
#     so dedupe survives restarts.
#   * Batch embeddings + reuse the Chroma collection (performance).
#   * Rich error handling — any stage failure becomes a typed IngestError.
# ============================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from app.config import settings
from app.services.chunker import chunk_document
from app.services.embedding import embed_texts
from app.services.pdf_reader import (
    PdfError,
    read_pdf,
    read_pdf_bytes,
)
from app.services.vectordb import get_vector_store
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


class IngestError(Exception):
    """Raised when ingestion fails at any stage."""


# ----------------------------------------------------------------------------
# Result object returned to the API layer.
# ----------------------------------------------------------------------------
@dataclass
class IngestResult:
    """Summary of what an ingestion run did."""
    document_name: str
    file_hash: str
    pages_extracted: int
    chunks_created: int
    chunks_stored: int
    skipped: bool               # True if we skipped because it was a duplicate
    message: str


# ----------------------------------------------------------------------------
# Manifest: remember which files (by hash) we've already ingested.
# ----------------------------------------------------------------------------
def _manifest_path() -> Path:
    """Location of the ingestion manifest JSON (next to the vector store)."""
    return settings.chroma_dir / "ingest_manifest.json"


def _load_manifest() -> dict[str, dict]:
    """Load the manifest {file_hash: {document_name, chunks, ...}} from disk."""
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # corrupt manifest shouldn't break ingestion
        log.warning("Could not read ingest manifest (%s). Starting fresh.", exc)
        return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    """Persist the manifest atomically (write temp then replace)."""
    path = _manifest_path()
    settings.ensure_directories()
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp, path)  # atomic on the same filesystem
    except Exception as exc:
        log.warning("Could not write ingest manifest (%s).", exc)


def is_already_ingested(file_hash: str) -> bool:
    """Return True if a file with this content hash was ingested before."""
    return file_hash in _load_manifest()


# ----------------------------------------------------------------------------
# The core pipeline (shared by file-path and upload entry points).
# ----------------------------------------------------------------------------
def _ingest_document(document, *, force: bool) -> IngestResult:
    """Chunk -> embed -> store a PdfDocument that has already been read.

    Args:
        document: a PdfDocument from pdf_reader.
        force:    if True, ingest even if the hash is already known.
    """
    manifest = _load_manifest()
    name = document.document_name
    file_hash = document.file_hash

    # --- Dedupe check ------------------------------------------------------
    if not force and file_hash in manifest:
        prev = manifest[file_hash]
        log.info("Skipping '%s' — identical content already ingested (%s chunks).",
                 name, prev.get("chunks_stored"))
        return IngestResult(
            document_name=name,
            file_hash=file_hash,
            pages_extracted=len(document.pages),
            chunks_created=0,
            chunks_stored=0,
            skipped=True,
            message=f"'{name}' already ingested (duplicate content). Use force=True to re-ingest.",
        )

    store = get_vector_store()

    # If forcing a re-ingest of the SAME document name, clear its old chunks
    # first so stale chunks (from an edited PDF) don't linger.
    if force:
        try:
            store.delete_document(name)
        except Exception as exc:
            log.warning("Could not pre-delete '%s' before force re-ingest: %s", name, exc)

    with timer(f"ingest:{name}", log):
        # --- Stage 2: chunk ------------------------------------------------
        chunks = chunk_document(document)
        if not chunks:
            raise IngestError(f"No chunks produced from '{name}'.")

        # --- Stage 3: embed (batched) --------------------------------------
        texts = [c.text for c in chunks]
        try:
            embeddings = embed_texts(texts)
        except Exception as exc:
            raise IngestError(f"Embedding failed for '{name}': {exc}") from exc

        # --- Stage 4: store ------------------------------------------------
        try:
            stored = store.upsert_chunks(chunks, embeddings)
        except Exception as exc:
            raise IngestError(f"Vector store write failed for '{name}': {exc}") from exc

    # --- Record in the manifest -------------------------------------------
    manifest[file_hash] = {
        "document_name": name,
        "pages_extracted": len(document.pages),
        "chunks_stored": stored,
    }
    _save_manifest(manifest)

    log.info("Ingested '%s': %d pages -> %d chunks stored.",
             name, len(document.pages), stored)

    return IngestResult(
        document_name=name,
        file_hash=file_hash,
        pages_extracted=len(document.pages),
        chunks_created=len(chunks),
        chunks_stored=stored,
        skipped=False,
        message=f"Ingested '{name}': {stored} chunks stored.",
    )


# ----------------------------------------------------------------------------
# Public entry points.
# ----------------------------------------------------------------------------
def ingest_pdf_file(path: str | os.PathLike[str], *,
                    password: str | None = None,
                    force: bool = False) -> IngestResult:
    """Ingest a single PDF from a filesystem path.

    Raises IngestError (wrapping any PdfError / embedding / store failure).
    """
    try:
        document = read_pdf(path, password=password)
    except PdfError as exc:
        raise IngestError(f"PDF read failed: {exc}") from exc
    return _ingest_document(document, force=force)


def ingest_pdf_bytes(data: bytes, document_name: str, *,
                     password: str | None = None,
                     force: bool = False) -> IngestResult:
    """Ingest a single PDF from raw bytes (e.g. a FastAPI upload)."""
    try:
        document = read_pdf_bytes(data, document_name, password=password)
    except PdfError as exc:
        raise IngestError(f"PDF read failed: {exc}") from exc
    return _ingest_document(document, force=force)


def ingest_directory(directory: str | os.PathLike[str] | None = None, *,
                     force: bool = False) -> list[IngestResult]:
    """Ingest EVERY .pdf in a directory (defaults to settings.data_dir).

    Handy for the initial load: drop PDFs into app/data/pdfs and call this.
    Per-file failures are logged and captured, but do not abort the batch.
    """
    folder = Path(directory) if directory else settings.data_dir
    if not folder.exists():
        raise IngestError(f"Ingestion directory not found: {folder}")

    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        log.warning("No PDF files found in '%s'.", folder)
        return []

    results: list[IngestResult] = []
    for pdf_path in pdf_paths:
        try:
            results.append(ingest_pdf_file(pdf_path, force=force))
        except IngestError as exc:
            log.error("Failed to ingest '%s': %s", pdf_path.name, exc)
            results.append(
                IngestResult(
                    document_name=pdf_path.name,
                    file_hash="",
                    pages_extracted=0,
                    chunks_created=0,
                    chunks_stored=0,
                    skipped=False,
                    message=f"ERROR: {exc}",
                )
            )
    return results


# ----------------------------------------------------------------------------
# Allow running as a script:  python -m app.services.ingest
# ----------------------------------------------------------------------------
def _main() -> None:
    """CLI: ingest every PDF sitting in app/data/pdfs."""
    log.info("Manual ingestion of all PDFs in '%s'...", settings.data_dir)
    results = ingest_directory()
    for r in results:
        print(json.dumps(asdict(r), indent=2))
    total = sum(r.chunks_stored for r in results)
    print(f"\nDONE. {len(results)} file(s), {total} chunks stored, "
          f"{get_vector_store().count()} total in collection.")


if __name__ == "__main__":
    _main()
