# ============================================================================
# config.py
# ----------------------------------------------------------------------------
# Centralised, type-safe configuration for the entire backend.
# Every tunable knob (paths, model names, chunk sizes, ports, timeouts) lives
# here so no other file hard-codes "magic" values.
#
# Think of this as appsettings.json + IOptions<T> from ASP.NET Core, but
# implemented with Pydantic's BaseSettings (validation + env overrides built in).
# ============================================================================

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve important directories relative to THIS file, not the current working
# directory. This makes the app run correctly no matter where you launch it.
BACKEND_DIR: Path = Path(__file__).resolve().parent          # .../insurance-rag/backend
PROJECT_ROOT: Path = BACKEND_DIR.parent                      # .../insurance-rag


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are resolved in this priority order (highest first):
      1. Environment variables (e.g. set OLLAMA_MODEL=llama3.1:70b)
      2. Values from a local .env file
      3. The defaults declared below
    """

    # --- Pydantic settings behaviour ---------------------------------------
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),   # optional local overrides
        env_file_encoding="utf-8",
        case_sensitive=False,                  # OLLAMA_MODEL == ollama_model
        extra="ignore",                        # ignore unrelated env vars
    )

    # --- Application metadata ----------------------------------------------
    app_name: str = "Insurance RAG API"
    app_version: str = "1.0.0"
    debug: bool = False

    # --- HTTP server -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS: which frontends may call this API. Angular dev server runs on 4200.
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:4200",
            "http://127.0.0.1:4200",
        ]
    )

    # --- Filesystem paths --------------------------------------------------
    # Where uploaded/sample PDFs live and where Chroma persists its data.
    # BACKEND_DIR resolves to .../insurance-rag/backend/app (config.py's folder),
    # so these line up with the project's app/ sub-structure.
    data_dir: Path = BACKEND_DIR / "data" / "pdfs"
    chroma_dir: Path = BACKEND_DIR / "database" / "chroma_db"
    logs_dir: Path = BACKEND_DIR / "logs"

    # --- Chunking ----------------------------------------------------------
    # 500-char chunks with 100-char overlap (per the spec).
    chunk_size: int = 500
    chunk_overlap: int = 100

    # --- Embeddings --------------------------------------------------------
    # Local sentence-transformers model. 384-dimensional output vectors.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    # How many chunks to embed per batch (performance vs. memory trade-off).
    embedding_batch_size: int = 32

    # --- Vector store (ChromaDB) -------------------------------------------
    collection_name: str = "insurance_claims"
    # Number of nearest chunks to retrieve for each question.
    top_k: int = 5

    # --- Ollama (local LLM) ------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_request_timeout: float = 120.0   # seconds; 8B model on CPU is slow
    ollama_temperature: float = 0.1         # low = factual, less "creative"
    ollama_num_ctx: int = 8192              # context window tokens

    # --- Deployment mode ---------------------------------------------------
    # "local" = Ollama + MiniLM + ChromaDB (default). "azure" = Azure OpenAI
    # (chat + embeddings) + ChromaDB. Set via env var BACKEND_MODE.
    backend_mode: str = "local"

    # --- Azure OpenAI (used only when backend_mode == "azure") -------------
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_chat_deployment: str = "gpt-5-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_embedding_dimension: int = 1536
    azure_max_completion_tokens: int = 2048
    # gpt-5 family are reasoning models: they spend tokens "thinking" before the
    # visible answer. "minimal" keeps that overhead (and cost/latency) low.
    # Set to "" to omit the parameter for non-reasoning models.
    azure_reasoning_effort: str = "minimal"

    # --- Safety / limits ---------------------------------------------------
    max_question_length: int = 2000         # reject absurdly long questions
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB PDF upload cap

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"

    def ensure_directories(self) -> None:
        """Create the folders the app needs, if they don't already exist.

        Called once at startup so ingestion/logging never fails because a
        directory is missing.
        """
        for directory in (self.data_dir, self.chroma_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def active_collection_name(self) -> str:
        """Vector-store collection name for the active backend.

        Azure embeddings are 1536-d while local MiniLM is 384-d, so we keep
        them in separate ChromaDB collections to avoid dimension clashes.
        """
        if self.backend_mode == "azure":
            return f"{self.collection_name}_azure"
        return self.collection_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings.

    lru_cache guarantees the .env file and environment are parsed only ONCE
    per process. Every module calls get_settings() and shares the same object,
    exactly like injecting IOptions<Settings> in ASP.NET Core.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


# A convenient module-level handle so simple scripts can do:
#   from config import settings
settings: Settings = get_settings()
