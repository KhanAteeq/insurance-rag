# ============================================================================
# app/services/pdf_reader.py
# ----------------------------------------------------------------------------
# STAGE 1 of the RAG pipeline: turn a binary PDF into clean, per-page text.
#
# We use PyMuPDF (imported as `fitz`) — a fast, pure-C PDF engine. This module
# does ONE job well: given a PDF path, return structured page text plus useful
# metadata, while gracefully handling the messy real world:
#     * missing file
#     * corrupt / non-PDF bytes
#     * encrypted (password-protected) PDF
#     * empty pages (scanned images with no text layer)
#     * completely empty documents
#
# NOTHING here knows about chunking, embeddings, or LLMs. Separation of
# concerns: this is purely "bytes -> text".
# ============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from app.utils.helpers import clean_text, file_sha1
from app.utils.logger import get_logger, timer

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Custom exceptions — so callers (ingest.py / the API) can react precisely
# instead of catching a generic Exception.
# ----------------------------------------------------------------------------
class PdfError(Exception):
    """Base class for all PDF-reading problems."""


class PdfNotFoundError(PdfError):
    """The file path does not exist."""


class PdfCorruptError(PdfError):
    """The bytes are not a valid/openable PDF."""


class PdfEncryptedError(PdfError):
    """The PDF is password-protected and we could not unlock it."""


class PdfEmptyError(PdfError):
    """The PDF opened fine but contains no extractable text on any page."""


# ----------------------------------------------------------------------------
# Data structures returned to the caller.
# ----------------------------------------------------------------------------
@dataclass
class PdfPage:
    """Text extracted from a single PDF page."""
    page_number: int          # 1-based, human-friendly
    text: str                 # cleaned plain text ("" if the page had none)
    char_count: int           # len(text), handy for logging / diagnostics


@dataclass
class PdfDocument:
    """The full result of reading one PDF file."""
    document_name: str                       # e.g. "surety-claims.pdf"
    source_path: str                         # absolute path on disk
    file_hash: str                           # sha1 of the raw bytes
    page_count: int                          # total pages in the PDF
    pages: list[PdfPage] = field(default_factory=list)  # only non-empty pages

    @property
    def total_characters(self) -> int:
        """Sum of characters across all extracted pages."""
        return sum(p.char_count for p in self.pages)

    @property
    def full_text(self) -> str:
        """All page text joined with blank lines (useful for debugging)."""
        return "\n\n".join(p.text for p in self.pages)


# ----------------------------------------------------------------------------
# The public function.
# ----------------------------------------------------------------------------
def read_pdf(path: str | os.PathLike[str], password: str | None = None) -> PdfDocument:
    """Read a PDF from disk and return structured, cleaned per-page text.

    Args:
        path: Path to the .pdf file.
        password: Optional password for encrypted PDFs.

    Returns:
        A PdfDocument containing only pages that actually had text.

    Raises:
        PdfNotFoundError:  path does not exist.
        PdfCorruptError:   file is not a valid PDF.
        PdfEncryptedError: PDF is encrypted and could not be unlocked.
        PdfEmptyError:     no page contained any extractable text.
    """
    pdf_path = Path(path)

    # --- 1. Existence check -------------------------------------------------
    if not pdf_path.exists():
        raise PdfNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise PdfNotFoundError(f"Path is not a file: {pdf_path}")

    document_name = pdf_path.name
    source_path = str(pdf_path.resolve())

    # Hash the raw bytes up-front so ingestion can dedupe identical files.
    file_hash = file_sha1(source_path)

    log.info("Reading PDF '%s' (%.1f KB)", document_name, pdf_path.stat().st_size / 1024)

    # --- 2. Open (catch corrupt files) -------------------------------------
    try:
        doc = fitz.open(source_path)
    except Exception as exc:  # PyMuPDF raises assorted errors for bad files
        raise PdfCorruptError(f"Could not open PDF '{document_name}': {exc}") from exc

    # Everything below runs inside try/finally so the document is always closed.
    try:
        # --- 3. Handle encryption ------------------------------------------
        if doc.needs_pass:
            # authenticate() returns a non-zero code on success.
            if not doc.authenticate(password or ""):
                raise PdfEncryptedError(
                    f"PDF '{document_name}' is password-protected and the "
                    f"supplied password was missing or incorrect."
                )
            log.info("Unlocked encrypted PDF '%s'.", document_name)

        page_count = doc.page_count

        # --- 4. Extract text page-by-page ----------------------------------
        pages: list[PdfPage] = []
        with timer(f"pdf_extract:{document_name}", log):
            for index in range(page_count):
                page = doc.load_page(index)          # 0-based internally
                raw = page.get_text("text")          # plain-text extraction
                cleaned = clean_text(raw)

                if not cleaned:
                    # Empty page (e.g. a scanned image with no text layer).
                    # We skip it but log it so operators know OCR may be needed.
                    log.debug("Page %d of '%s' had no extractable text (skipped).",
                              index + 1, document_name)
                    continue

                pages.append(
                    PdfPage(
                        page_number=index + 1,       # store 1-based
                        text=cleaned,
                        char_count=len(cleaned),
                    )
                )
    finally:
        doc.close()

    # --- 5. Guard against a totally empty document -------------------------
    if not pages:
        raise PdfEmptyError(
            f"PDF '{document_name}' opened but contained no extractable text "
            f"on any of its {page_count} page(s). It may be a scanned image "
            f"needing OCR."
        )

    result = PdfDocument(
        document_name=document_name,
        source_path=source_path,
        file_hash=file_hash,
        page_count=page_count,
        pages=pages,
    )

    log.info(
        "Extracted %d/%d non-empty pages, %d characters from '%s'.",
        len(result.pages), page_count, result.total_characters, document_name,
    )
    return result


def read_pdf_bytes(data: bytes, document_name: str,
                   password: str | None = None) -> PdfDocument:
    """Read a PDF directly from in-memory bytes (e.g. a FastAPI upload).

    Writes nothing to disk permanently — opens the byte stream with PyMuPDF.
    The file_hash is computed from the provided bytes.

    Raises the same PdfError subclasses as read_pdf().
    """
    import hashlib

    if not data:
        raise PdfEmptyError(f"Uploaded PDF '{document_name}' contained 0 bytes.")

    file_hash = hashlib.sha1(data).hexdigest()

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfCorruptError(f"Could not open uploaded PDF '{document_name}': {exc}") from exc

    try:
        if doc.needs_pass:
            if not doc.authenticate(password or ""):
                raise PdfEncryptedError(
                    f"Uploaded PDF '{document_name}' is password-protected and "
                    f"the supplied password was missing or incorrect."
                )

        page_count = doc.page_count
        pages: list[PdfPage] = []
        with timer(f"pdf_extract:{document_name}", log):
            for index in range(page_count):
                page = doc.load_page(index)
                cleaned = clean_text(page.get_text("text"))
                if not cleaned:
                    continue
                pages.append(
                    PdfPage(page_number=index + 1, text=cleaned, char_count=len(cleaned))
                )
    finally:
        doc.close()

    if not pages:
        raise PdfEmptyError(
            f"Uploaded PDF '{document_name}' contained no extractable text."
        )

    return PdfDocument(
        document_name=document_name,
        source_path=f"<upload:{document_name}>",
        file_hash=file_hash,
        page_count=page_count,
        pages=pages,
    )
