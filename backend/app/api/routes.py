# ============================================================================
# app/api/routes.py
# ----------------------------------------------------------------------------
# The HTTP surface of the backend — a FastAPI APIRouter mounted under /api.
#
# Endpoints:
#   GET  /api/health              -> service + Ollama + vector-store status
#   POST /api/ask                 -> ask a question (blocking, full JSON answer)
#   POST /api/ask/stream          -> ask a question (streaming NDJSON tokens)
#   POST /api/ingest              -> upload a PDF and ingest it
#   POST /api/ingest/directory    -> ingest every PDF in the server data dir
#   GET  /api/documents           -> list indexed documents + counts
#   GET  /api/chunks              -> sample stored chunks (debug/inspect)
#   DELETE /api/documents/{name}  -> remove one document's chunks
#   POST /api/reset               -> wipe the entire vector store (danger)
#
# Every handler is thin: it validates input, calls a service, and maps typed
# domain errors to clean HTTP status codes. No business logic lives here.
# ============================================================================

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.models import (
    AskRequest,
    AskResponse,
    ChunkPreview,
    ChunksResponse,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestDirectoryResponse,
    IngestResponse,
    OllamaHealth,
)
from app.config import settings
from app.services import llm
from app.services.embedding import EmbeddingError
from app.services.ingest import (
    IngestError,
    IngestResult,
    ingest_directory,
    ingest_pdf_bytes,
)
from app.services.rag import RagError, get_rag_service
from app.services.vectordb import VectorStoreError, get_vector_store
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])


# ----------------------------------------------------------------------------
# Helper: convert an internal IngestResult -> API IngestResponse.
# ----------------------------------------------------------------------------
def _to_ingest_response(r: IngestResult) -> IngestResponse:
    return IngestResponse(
        document_name=r.document_name,
        file_hash=r.file_hash,
        pages_extracted=r.pages_extracted,
        chunks_created=r.chunks_created,
        chunks_stored=r.chunks_stored,
        skipped=r.skipped,
        message=r.message,
    )


# ----------------------------------------------------------------------------
# GET /api/health
# ----------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    """Report overall health: vector-store size + Ollama/model availability."""
    llm_status = llm.health()
    try:
        total_chunks = get_vector_store().count()
    except Exception as exc:
        log.warning("Health: vector store count failed: %s", exc)
        total_chunks = -1

    overall = "ok" if llm_status.get("reachable") and llm_status.get("model_available") \
        else "degraded"

    return HealthResponse(
        status=overall,
        app_name=settings.app_name,
        app_version=settings.app_version,
        total_chunks=total_chunks,
        embedding_model=settings.embedding_model_name,
        llm=OllamaHealth(
            reachable=bool(llm_status.get("reachable")),
            model=str(llm_status.get("model")),
            model_available=bool(llm_status.get("model_available")),
            models=list(llm_status.get("models", [])),
        ),
    )


# ----------------------------------------------------------------------------
# POST /api/ask   (blocking)
# ----------------------------------------------------------------------------
@router.post("/ask", response_model=AskResponse, summary="Ask a question")
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the ingested documents (blocking).

    Maps domain errors to HTTP status codes:
        400 -> invalid question (empty / too long)
        503 -> Ollama unavailable or model not pulled
        504 -> LLM timed out
        500 -> any other pipeline failure
    """
    service = get_rag_service()
    try:
        result = service.answer(
            request.question,
            top_k=request.top_k,
            document_name=request.document_name,
        )
    except ValueError as exc:  # from sanitize_question
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=str(exc)) from exc
    except llm.LLMUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=str(exc)) from exc
    except llm.LLMModelMissingError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=str(exc)) from exc
    except llm.LLMTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail=str(exc)) from exc
    except (RagError, EmbeddingError, VectorStoreError, llm.LLMError) as exc:
        log.exception("RAG /ask failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    return AskResponse(**result.as_dict())


# ----------------------------------------------------------------------------
# POST /api/ask/stream   (streaming NDJSON)
# ----------------------------------------------------------------------------
@router.post("/ask/stream", summary="Ask a question (streaming)")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Answer a question, streaming newline-delimited JSON (NDJSON) events.

    Each line is a JSON object of one of these shapes:
        {"type":"sources","sources":[...],"grounded":true}
        {"type":"token","token":"..."}
        {"type":"done","timings":{...},"grounded":true,"model":"..."}
        {"type":"error","message":"..."}

    The Angular client reads the body incrementally to render a live answer.
    """
    service = get_rag_service()

    def event_stream() -> Iterator[bytes]:
        try:
            for event in service.answer_stream(
                request.question,
                top_k=request.top_k,
                document_name=request.document_name,
            ):
                yield (json.dumps(event) + "\n").encode("utf-8")
        except Exception as exc:  # last-resort guard inside the generator
            log.exception("Streaming /ask failed.")
            yield (json.dumps({"type": "error", "message": str(exc)}) + "\n").encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------------------------------------------------------
# POST /api/ingest   (upload a PDF)
# ----------------------------------------------------------------------------
@router.post("/ingest", response_model=IngestResponse, summary="Ingest an uploaded PDF")
async def ingest_upload(
    file: UploadFile = File(..., description="A .pdf file to ingest."),
    force: bool = Query(False, description="Re-ingest even if content is a duplicate."),
) -> IngestResponse:
    """Upload a single PDF, chunk + embed + store it in the vector database."""
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Only .pdf files are accepted.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024*1024)} MB limit.",
        )

    try:
        result = ingest_pdf_bytes(data, filename, force=force)
    except IngestError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Ingest upload failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    return _to_ingest_response(result)


# ----------------------------------------------------------------------------
# POST /api/ingest/directory   (ingest all PDFs on the server)
# ----------------------------------------------------------------------------
@router.post("/ingest/directory", response_model=IngestDirectoryResponse,
             summary="Ingest all PDFs in the server data directory")
def ingest_all(
    force: bool = Query(False, description="Re-ingest even duplicates."),
) -> IngestDirectoryResponse:
    """Ingest every PDF sitting in the server's configured data directory."""
    try:
        results = ingest_directory(force=force)
    except IngestError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Directory ingest failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    responses = [_to_ingest_response(r) for r in results]
    return IngestDirectoryResponse(
        total_files=len(responses),
        total_chunks_stored=sum(r.chunks_stored for r in responses),
        results=responses,
    )


# ----------------------------------------------------------------------------
# GET /api/documents
# ----------------------------------------------------------------------------
@router.get("/documents", response_model=DocumentsResponse,
            summary="List indexed documents")
def list_documents() -> DocumentsResponse:
    """Return every document currently in the vector store and its chunk count."""
    try:
        store = get_vector_store()
        docs = store.list_documents()
        total_chunks = store.count()
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    return DocumentsResponse(
        total_documents=len(docs),
        total_chunks=total_chunks,
        documents=[DocumentInfo(**d) for d in docs],
    )


# ----------------------------------------------------------------------------
# GET /api/chunks
# ----------------------------------------------------------------------------
@router.get("/chunks", response_model=ChunksResponse,
            summary="Sample stored chunks")
def sample_chunks(
    limit: int = Query(10, ge=1, le=100, description="How many chunks to preview."),
) -> ChunksResponse:
    """Return a small sample of stored chunks for inspection/debugging."""
    try:
        store = get_vector_store()
        sample = store.peek_chunks(limit=limit)
        total = store.count()
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc

    return ChunksResponse(
        total_chunks=total,
        sample=[ChunkPreview(**c) for c in sample],
    )


# ----------------------------------------------------------------------------
# DELETE /api/documents/{document_name}
# ----------------------------------------------------------------------------
@router.delete("/documents/{document_name}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a document's chunks")
def delete_document(document_name: str) -> Response:
    """Remove every chunk belonging to a single source document."""
    try:
        get_vector_store().delete_document(document_name)
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ----------------------------------------------------------------------------
# POST /api/reset   (danger)
# ----------------------------------------------------------------------------
@router.post("/reset", status_code=status.HTTP_204_NO_CONTENT,
             summary="Reset the entire vector store")
def reset_store() -> Response:
    """Delete the whole collection and recreate it empty. Destructive."""
    try:
        get_vector_store().reset()
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
