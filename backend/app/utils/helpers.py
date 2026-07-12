# Helper utility functions
# ============================================================================
# app/utils/helpers.py
# ----------------------------------------------------------------------------
# Pure helper functions shared across services:
#   * Input sanitisation + validation (security layer).
#   * Stable, deterministic chunk-ID generation (avoids duplicate ingestion).
#   * Streamed file hashing (detect already-ingested PDFs).
#
# These are dependency-light, side-effect-free utilities — easy to unit test.
# ============================================================================

from __future__ import annotations

import hashlib
import re

from app.config import settings


# ----------------------------------------------------------------------------
# INPUT SANITISATION / VALIDATION (security)
# ----------------------------------------------------------------------------
# Collapse any run of whitespace (spaces, tabs, newlines) into a single space.
_WHITESPACE_RE = re.compile(r"\s+")
# Strip ASCII control characters (except normal whitespace) that can corrupt
# logs or terminals and are never legitimate in a user question.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalise raw extracted/user text.

    * Removes control characters.
    * Collapses repeated whitespace.
    * Trims leading/trailing spaces.
    Used both on PDF-extracted text and on user questions.
    """
    if not text:
        return ""
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def sanitize_question(question: str) -> str:
    """Validate + clean a user question before it enters the RAG pipeline.

    Raises ValueError on empty or over-length input so the API layer can
    return a clean HTTP 400 instead of failing deep inside the LLM call.
    """
    if question is None:
        raise ValueError("Question must not be null.")

    cleaned = clean_text(question)

    if not cleaned:
        raise ValueError("Question must not be empty.")

    if len(cleaned) > settings.max_question_length:
        raise ValueError(
            f"Question too long ({len(cleaned)} chars). "
            f"Maximum allowed is {settings.max_question_length}."
        )

    return cleaned


# ----------------------------------------------------------------------------
# STABLE ID GENERATION (avoid duplicate ingestion)
# ----------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Turn a document name into a filesystem/ID-safe slug.

    Keeps letters, digits, dash and underscore; replaces everything else
    with underscores. Prevents weird characters from breaking IDs.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "doc"


def make_chunk_id(document_name: str, page_number: int, chunk_number: int) -> str:
    """Create a deterministic, collision-resistant ID for a chunk.

    Same (document, page, chunk) -> same ID every time. This lets the vector
    store UPSERT: re-ingesting an unchanged PDF overwrites the same IDs
    instead of creating duplicates.
    """
    raw = f"{document_name}::page{page_number}::chunk{chunk_number}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{slugify(document_name)}-{page_number}-{chunk_number}-{digest}"


def file_sha1(path: str) -> str:
    """Compute a SHA-1 hash of a file's bytes (streamed, memory-safe).

    Used to detect whether a PDF has already been ingested unchanged, so we
    can skip re-processing identical files.
    """
    hasher = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()
