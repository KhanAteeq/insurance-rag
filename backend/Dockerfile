# ============================================================================
# Backend container image for the Insurance RAG API (FastAPI + Uvicorn).
# Build:  docker build -t insurance-rag-api ./backend
# Run:    docker run -p 8000:8000 --env-file backend/.env insurance-rag-api
# ============================================================================
FROM python:3.12-slim

# System deps kept minimal; PyMuPDF ships manylinux wheels so no build tools needed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code.
COPY app ./app

# Run as a non-root user (security best practice).
RUN useradd --create-home appuser \
    && mkdir -p /app/app/database/chroma_db /app/app/data/pdfs /app/app/logs \
    && chown -R appuser:appuser /app /home/appuser 2>/dev/null || true
USER appuser

# Azure Container Apps sends traffic to the port in $PORT (default 8000 here).
ENV PORT=8000
EXPOSE 8000

# Bind to 0.0.0.0 and the platform-provided port.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
