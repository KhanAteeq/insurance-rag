# ============================================================================
# app/main.py
# ----------------------------------------------------------------------------
# The FastAPI application entry point — the "Program.cs" of the backend.
#
# Responsibilities:
#   * Build the FastAPI app (title, version, docs).
#   * Configure CORS so the Angular dev server (localhost:4200) can call us.
#   * Register the /api router (all endpoints).
#   * Install global exception handlers for a uniform error envelope.
#   * Run a startup "lifespan" that pre-warms the embedding model and the LLM,
#     and auto-ingests any PDFs sitting in the data directory.
#
# Run it with:
#     uvicorn app.main:app --reload            (from the backend/ folder)
#   or simply:
#     python -m app.main
# ============================================================================

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.models import ErrorResponse
from app.api.routes import router
from app.config import settings
from app.services import embedding, llm
from app.services.ingest import ingest_directory
from app.utils.logger import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Lifespan: run once on startup and once on shutdown.
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook.

    Startup:
      * ensure directories exist,
      * eagerly load the embedding model (first request is then fast),
      * warm the LLM (best-effort; skipped if Ollama is down),
      * auto-ingest any PDFs already sitting in the data directory.
    """
    log.info("Starting %s v%s ...", settings.app_name, settings.app_version)
    settings.ensure_directories()

    # Pre-load the embedding model so the first /ask isn't penalised.
    try:
        embedding.warm_up()
    except Exception as exc:
        log.warning("Embedding warm-up failed (will lazy-load later): %s", exc)

    # Warm the LLM (never fatal — Ollama may be started after the API).
    try:
        llm.warm_up()
    except Exception as exc:
        log.warning("LLM warm-up skipped: %s", exc)

    # Auto-ingest any PDFs already present so the app is useful immediately.
    # On Azure App Service (Free tier) we SKIP this so the container boots fast
    # and passes the startup probe; populate on demand via /api/ingest/directory.
    import os
    if os.getenv("WEBSITE_SITE_NAME"):
        log.info("Detected App Service — skipping startup auto-ingest "
                 "(call POST /api/ingest/directory once to populate).")
    else:
        try:
            results = ingest_directory()
            stored = sum(r.chunks_stored for r in results)
            if results:
                log.info("Startup ingest: %d file(s), %d new chunk(s) stored.",
                         len(results), stored)
        except Exception as exc:
            log.warning("Startup auto-ingest skipped: %s", exc)

    log.info("%s is ready at http://%s:%d", settings.app_name,
             settings.host, settings.port)
    try:
        yield
    finally:
        log.info("Shutting down %s.", settings.app_name)


# ----------------------------------------------------------------------------
# Build the application.
# ----------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Application factory — construct and configure the FastAPI instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Local Retrieval-Augmented Generation API for insurance claim "
            "documents. Runs entirely offline: PyMuPDF + SentenceTransformers "
            "+ ChromaDB + Ollama (llama3.1:8b)."
        ),
        lifespan=lifespan,
    )

    # --- CORS: allow the Angular frontend to call the API. ----------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ------------------------------------------------------------
    app.include_router(router)

    # --- Global exception handlers (uniform error envelope) ---------------
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request,
                                  exc: RequestValidationError) -> JSONResponse:
        """Return a clean 422 for request-body validation failures."""
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail=str(exc.errors()),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all so an unexpected error never leaks a stack trace to clients."""
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail="An unexpected error occurred. Please check server logs.",
            ).model_dump(),
        )

    # --- Root + simple liveness probe -------------------------------------
    @app.get("/", tags=["meta"], summary="Service banner")
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


# The ASGI application object uvicorn looks for: `app.main:app`.
app = create_app()


# ----------------------------------------------------------------------------
# Allow `python -m app.main` to launch the server directly.
# ----------------------------------------------------------------------------
def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
