# ============================================================================
# app/api/models.py
# ----------------------------------------------------------------------------
# The API's DATA CONTRACT: Pydantic v2 schemas for every request and response.
#
# These models are the boundary between the outside world (Angular / curl) and
# the internal service objects. FastAPI uses them to:
#   * validate + coerce incoming JSON (bad input -> automatic HTTP 422),
#   * serialise outgoing objects to JSON,
#   * auto-generate the OpenAPI / Swagger docs at /docs.
#
# Keeping them in one file gives a single, reviewable view of the whole API
# surface — like a set of C# DTOs / records.
# ============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------------------
# /ask  — ask a question against the ingested documents
# ----------------------------------------------------------------------------
class AskRequest(BaseModel):
    """Body for POST /api/ask."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The natural-language question to answer from the documents.",
        examples=["What is the bond number for claim CLM-2026-0042?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="How many chunks to retrieve. Defaults to the server config.",
    )
    document_name: str | None = Field(
        default=None,
        description="Optional: restrict retrieval to a single source document.",
    )


class SourceModel(BaseModel):
    """One cited source chunk returned alongside an answer."""
    id: str
    document_name: str
    page_number: int
    chunk_number: int
    section: str
    score: float = Field(description="Similarity in [0, 1]; higher is better.")


class TimingModel(BaseModel):
    """Per-stage latency breakdown, in milliseconds."""
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class AskResponse(BaseModel):
    """Body returned by POST /api/ask."""
    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    sources: list[SourceModel] = Field(default_factory=list)
    model: str
    timings: TimingModel
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    grounded: bool = Field(
        description="True if the answer was grounded in retrieved context."
    )


# ----------------------------------------------------------------------------
# /ingest  — add PDFs to the knowledge base
# ----------------------------------------------------------------------------
class IngestResponse(BaseModel):
    """Result of ingesting a single PDF (upload or file path)."""
    document_name: str
    file_hash: str
    pages_extracted: int
    chunks_created: int
    chunks_stored: int
    skipped: bool = Field(
        description="True if skipped because identical content was already ingested."
    )
    message: str


class IngestDirectoryResponse(BaseModel):
    """Result of ingesting every PDF in the server's data directory."""
    total_files: int
    total_chunks_stored: int
    results: list[IngestResponse]


# ----------------------------------------------------------------------------
# /documents  &  /chunks  — inspect the knowledge base
# ----------------------------------------------------------------------------
class DocumentInfo(BaseModel):
    """Per-document stats in the vector store."""
    document_name: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """List of documents currently indexed."""
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]


class ChunkPreview(BaseModel):
    """A truncated preview of one stored chunk."""
    id: str
    text: str
    metadata: dict


class ChunksResponse(BaseModel):
    """A small sample of stored chunks (for debugging / inspection)."""
    total_chunks: int
    sample: list[ChunkPreview]


# ----------------------------------------------------------------------------
# /health  — service + dependency status
# ----------------------------------------------------------------------------
class OllamaHealth(BaseModel):
    """Health of the local Ollama LLM dependency."""
    model_config = ConfigDict(protected_namespaces=())

    reachable: bool
    model: str
    model_available: bool
    models: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Overall service health."""
    status: str = Field(description="'ok' or 'degraded'.")
    app_name: str
    app_version: str
    total_chunks: int
    embedding_model: str
    llm: OllamaHealth


# ----------------------------------------------------------------------------
# Generic error envelope (used by exception handlers)
# ----------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Uniform error body returned for handled failures."""
    error: str = Field(description="Short machine-readable error type.")
    detail: str = Field(description="Human-readable explanation.")
