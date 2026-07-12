# Insurance RAG — Local Retrieval-Augmented Generation

A production-quality, **fully local** RAG system for insurance claim documents.
No cloud, no OpenAI, no Azure, no AWS. Everything runs on your machine.

| Layer         | Technology                                            |
| ------------- | ----------------------------------------------------- |
| PDF reading   | PyMuPDF (`fitz`)                                       |
| Chunking      | Manual recursive character splitter (500 / 100)       |
| Embeddings    | SentenceTransformers `all-MiniLM-L6-v2` (384-dim)     |
| Vector store  | ChromaDB (persistent, cosine)                         |
| LLM           | Ollama + `llama3.1:8b`                                 |
| API           | FastAPI + Uvicorn                                      |
| Frontend      | Angular 18 + Angular Material                          |

The full pipeline is built **manually** (no LangChain) so every stage is
inspectable: `pdf_reader → chunker → embedding → vectordb → retriever →
prompt_builder → llm → rag`.

---

## 1. Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm** (for the Angular frontend)
- **Ollama** — https://ollama.com/download

Pull the model once:

```powershell
ollama pull llama3.1:8b
ollama serve   # usually already running as a background service
```

---

## 2. Backend setup

```powershell
cd insurance-rag\backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Optional: copy the example env and tweak values
Copy-Item .env.example .env
```

Place any PDF you want indexed into:

```
backend\app\data\pdfs\
```

Run the API (from the `backend` folder):

```powershell
uvicorn app.main:app --reload
```

- Swagger docs: http://localhost:8000/docs
- Health:       http://localhost:8000/api/health

On startup the server pre-loads the embedding model, warms the LLM, and
auto-ingests every PDF found in `app/data/pdfs`.

### Manual ingestion (optional)

```powershell
python -m app.services.ingest
```

---

## 3. API endpoints

| Method | Path                         | Description                              |
| ------ | ---------------------------- | ---------------------------------------- |
| GET    | `/api/health`                | Service + Ollama + vector-store status   |
| POST   | `/api/ask`                   | Ask a question (blocking JSON answer)    |
| POST   | `/api/ask/stream`            | Ask a question (streaming NDJSON tokens) |
| POST   | `/api/ingest`                | Upload and ingest a single PDF           |
| POST   | `/api/ingest/directory`      | Ingest every PDF in the data directory   |
| GET    | `/api/documents`             | List indexed documents + chunk counts    |
| GET    | `/api/chunks?limit=10`       | Sample stored chunks (debug)             |
| DELETE | `/api/documents/{name}`      | Delete one document's chunks             |
| POST   | `/api/reset`                 | Wipe the entire vector store (danger)    |

### Example: ask a question

```powershell
curl -X POST http://localhost:8000/api/ask `
  -H "Content-Type: application/json" `
  -d '{ "question": "What is the bond number for claim CLM-2026-0042?" }'
```

---

## 4. Frontend setup

```powershell
cd insurance-rag\frontend

npm install
npm start           # ng serve on http://localhost:4200
```

The Angular app talks to the backend at `http://localhost:8000` (configured in
`src/environments/environment.ts`). Make sure the backend is running first.

---

## 5. Project layout

```
insurance-rag/
├── backend/
│   ├── app/
│   │   ├── api/            routes.py, models.py
│   │   ├── services/       pdf_reader, chunker, embedding, vectordb,
│   │   │                   retriever, prompt_builder, llm, rag, ingest
│   │   ├── database/chroma_db/   (persistent vector store)
│   │   ├── data/pdfs/            (drop PDFs here)
│   │   ├── utils/          logger.py, helpers.py
│   │   ├── config.py
│   │   └── main.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
└── frontend/               Angular 18 + Material chat UI
```

---

## 6. How the RAG pipeline works

1. **Read** — PyMuPDF extracts clean, per-page text from each PDF.
2. **Chunk** — recursive character splitter cuts text into ~500-char chunks
   with 100-char overlap, tagging page/section metadata.
3. **Embed** — `all-MiniLM-L6-v2` turns each chunk into a normalised 384-dim
   vector.
4. **Store** — vectors + text + metadata are upserted into ChromaDB (cosine).
5. **Retrieve** — the question is embedded and the top-k nearest chunks are
   fetched and score-filtered.
6. **Prompt** — a strict, grounded prompt fences the retrieved context and
   forbids using outside knowledge (anti-hallucination + anti-injection).
7. **Generate** — Ollama (`llama3.1:8b`) produces the answer, cited to sources.

If retrieval finds nothing relevant, the service short-circuits and returns
`"I don't know based on the provided documents."` without calling the LLM.

---

## 7. Troubleshooting

| Symptom                                  | Fix                                              |
| ---------------------------------------- | ------------------------------------------------ |
| `/api/health` shows `reachable: false`   | Start Ollama: `ollama serve`                     |
| `model_available: false`                 | `ollama pull llama3.1:8b`                         |
| First `/ask` is slow                     | Model weights load on first use (then cached)    |
| No answers / empty results               | Ingest a PDF first (drop into `app/data/pdfs`)   |
| CORS error in the browser                | Confirm frontend origin is in `CORS_ALLOW_ORIGINS` |
