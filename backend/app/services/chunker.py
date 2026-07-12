# Text chunking service
# ============================================================================
# app/services/chunker.py
# ----------------------------------------------------------------------------
# STAGE 2 of the RAG pipeline: split each page's text into small, overlapping
# "chunks" and attach rich metadata to every chunk.
#
# WHY CHUNK AT ALL?
#   * Embedding models have a limited input window and produce ONE vector per
#     input. If you embed a whole 2,000-word page as a single vector, that
#     vector is a blurry "average" of many topics -> poor retrieval.
#   * The LLM has a limited context window. We must retrieve only the few
#     hundred characters that actually answer the question, not entire pages.
#   * Smaller, focused chunks => sharper embeddings => more precise retrieval.
#
# STRATEGY: manual RECURSIVE character chunking.
#   Try to split on the most natural boundary first (paragraphs), then
#   sentences, then words, then raw characters — so we cut at meaningful seams
#   instead of slicing words in half. Target size 500 chars, overlap 100 chars.
#
# OVERLAP: consecutive chunks share ~100 characters so a fact sitting on a
# chunk boundary is not lost ("the deductible is | $5,000" split across two
# chunks would otherwise be un-retrievable).
#
# No LangChain — we implement the recursive splitter ourselves.
# ============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.services.pdf_reader import PdfDocument
from app.utils.helpers import make_chunk_id
from app.utils.logger import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Data structure: one chunk + all the metadata the spec requires.
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance metadata."""
    id: str                       # stable, deterministic (from make_chunk_id)
    text: str                     # the chunk content (<= chunk_size-ish)
    document_name: str            # which PDF this came from
    page_number: int              # which page (1-based)
    chunk_number: int             # sequential index within the WHOLE document
    section: str                  # best-guess heading/section label
    char_count: int               # len(text)

    def metadata(self) -> dict:
        """Return a Chroma-friendly metadata dict (all values are scalars).

        ChromaDB only accepts str/int/float/bool metadata values, so we keep
        this flat and primitive.
        """
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_number": self.chunk_number,
            "section": self.section,
            "char_count": self.char_count,
        }


# ----------------------------------------------------------------------------
# Section detection (best-effort).
# ----------------------------------------------------------------------------
# A "section" heading heuristic: a short line that looks like a title, e.g.
#   "CLAIM SUMMARY", "3. Coverage Details", "Section B - Loss Information".
# This is metadata only; it never blocks chunking.
_SECTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:section|part|article|clause)\s+[\w.\-]+"   # "Section B", "Part 3"
    r"|\d+(?:\.\d+)*\s+[A-Z][^\n]{2,60}"            # "3.1 Coverage Details"
    r"|[A-Z][A-Z0-9 &/\-]{3,60}"                    # "CLAIM SUMMARY" (all caps)
    r")\s*$",
    re.MULTILINE,
)


def _detect_section(page_text: str, fallback: str) -> str:
    """Return the first heading-like line on a page, else the fallback.

    Cheap heuristic: insurance PDFs usually start a page/section with a short
    upper-case or numbered heading. Good enough to enrich retrieval; not a
    full document-structure parser.
    """
    match = _SECTION_RE.search(page_text)
    if match:
        return match.group(0).strip()[:80]
    return fallback


# ----------------------------------------------------------------------------
# The recursive character splitter (implemented manually).
# ----------------------------------------------------------------------------
# Ordered list of separators from "most natural" to "last resort".
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into <= chunk_size pieces with `overlap` chars of context.

    Algorithm (recursive character splitting):
      1. If the text already fits, return it as a single chunk.
      2. Otherwise pick the FIRST separator that actually occurs in the text
         and split on it into "atoms" (paragraphs, then lines, sentences...).
      3. Greedily pack atoms into a buffer until adding the next one would
         exceed chunk_size; emit the buffer as a chunk.
      4. Start the next buffer with the tail `overlap` characters of the
         previous chunk so context carries over.
      5. If a single atom is itself larger than chunk_size, recurse into it
         with the NEXT (finer) separator.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Choose the finest-but-present separator, walking from coarse to fine.
    separator = _SEPARATORS[-1]  # default: "" (character split)
    for sep in _SEPARATORS:
        if sep == "":
            separator = ""
            break
        if sep in text:
            separator = sep
            break

    # Break the text into atoms using the chosen separator.
    if separator == "":
        # Hard character split: chop into chunk_size windows directly.
        atoms = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        parts = text.split(separator)
        # Re-attach the separator (except for the last part) so we don't lose
        # the punctuation/spacing that gives text meaning.
        atoms = [p + separator for p in parts[:-1]] + [parts[-1]]

    chunks: list[str] = []
    buffer = ""

    for atom in atoms:
        # If a single atom is too big on its own, recurse with a finer split.
        if len(atom) > chunk_size:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            sub_index = _SEPARATORS.index(separator) + 1 if separator in _SEPARATORS else len(_SEPARATORS)
            finer = _SEPARATORS[sub_index:] or [""]
            chunks.extend(_split_with_separators(atom, chunk_size, overlap, finer))
            continue

        if len(buffer) + len(atom) <= chunk_size:
            buffer += atom
        else:
            # Emit the current buffer, then start a new one carrying overlap.
            if buffer.strip():
                chunks.append(buffer.strip())
            tail = buffer[-overlap:] if overlap > 0 else ""
            buffer = tail + atom

    if buffer.strip():
        chunks.append(buffer.strip())

    return _apply_overlap_dedupe(chunks)


def _split_with_separators(text: str, chunk_size: int, overlap: int,
                           separators: list[str]) -> list[str]:
    """Helper: split using a specific (finer) separator list. Recurses.

    Used when a single atom is larger than chunk_size and must be broken down
    using progressively finer separators.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    separator = separators[-1]
    for sep in separators:
        if sep == "" or sep in text:
            separator = sep
            break

    if separator == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    atoms = [p + separator for p in parts[:-1]] + [parts[-1]]

    chunks: list[str] = []
    buffer = ""
    for atom in atoms:
        if len(atom) > chunk_size:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            idx = separators.index(separator) + 1
            chunks.extend(_split_with_separators(atom, chunk_size, overlap,
                                                 separators[idx:] or [""]))
            continue
        if len(buffer) + len(atom) <= chunk_size:
            buffer += atom
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            tail = buffer[-overlap:] if overlap > 0 else ""
            buffer = tail + atom
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def _apply_overlap_dedupe(chunks: list[str]) -> list[str]:
    """Remove accidental exact-duplicate neighbouring chunks.

    Overlap logic can occasionally produce an identical neighbour when the
    text is short; drop consecutive duplicates to keep the index clean.
    """
    result: list[str] = []
    for c in chunks:
        if not result or result[-1] != c:
            result.append(c)
    return result


# ----------------------------------------------------------------------------
# Public API: chunk a whole PdfDocument.
# ----------------------------------------------------------------------------
def chunk_document(
    document: PdfDocument,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split every page of a PdfDocument into overlapping Chunks with metadata.

    Args:
        document:   the PdfDocument produced by pdf_reader.read_pdf().
        chunk_size: max characters per chunk (defaults to settings.chunk_size).
        overlap:    overlap characters between chunks (defaults to config).

    Returns:
        A flat list of Chunk objects across all pages, numbered sequentially.
    """
    size = chunk_size or settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap

    if ov >= size:
        raise ValueError(
            f"chunk_overlap ({ov}) must be smaller than chunk_size ({size})."
        )

    chunks: list[Chunk] = []
    running_index = 0  # sequential chunk number across the WHOLE document

    for page in document.pages:
        section = _detect_section(page.text, fallback=f"Page {page.page_number}")
        pieces = _split_text(page.text, size, ov)

        for piece in pieces:
            chunk_id = make_chunk_id(document.document_name, page.page_number,
                                     running_index)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    document_name=document.document_name,
                    page_number=page.page_number,
                    chunk_number=running_index,
                    section=section,
                    char_count=len(piece),
                )
            )
            running_index += 1

    log.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d).",
        document.document_name, len(chunks), size, ov,
    )
    return chunks
