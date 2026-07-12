# Insurance RAG — 5-Page Interview Cheat Sheet

> Local, offline RAG over insurance claim PDFs. Stack: **Angular 18 + FastAPI + PyMuPDF + SentenceTransformers (all-MiniLM-L6-v2) + ChromaDB + Ollama (llama3.1:8b)**. Built manually (no LangChain).

---

## 1. ARCHITECTURE (one glance)

```
┌────────────┐   HTTP/JSON + NDJSON stream   ┌──────────────────────────────┐
│ Angular 18 │ ────────────────────────────► │ FastAPI (uvicorn)  :8001     │
│ + Material │ ◄──────────────────────────── │  api/routes.py               │
└────────────┘                               └──────────────┬───────────────┘
      UI: chat.component                                     │ calls
      svc: chat.service (HttpClient + fetch stream)          ▼
                                              ┌──────────────────────────────┐
                                              │ rag.py (orchestrator)        │
                                              └───┬───────────┬──────────┬────┘
                        retrieve │        prompt  │      generate │
                                 ▼                ▼               ▼
                        retriever.py       prompt_builder.py     llm.py
                             │                                    │ HTTP
                     embed   │ search                             ▼
                             ▼                                  Ollama :11434
                    embedding.py → vectordb.py (ChromaDB, cosine)  llama3.1:8b
```

**Two paths:**
- **Write (ingest):** `pdf_reader → chunker → embedding → vectordb.upsert` (orchestrated by `ingest.py`).
- **Read (ask):** `sanitize → retriever(embed+search) → prompt_builder → llm → RagAnswer`.

---

## 2. TECHNOLOGY STACK

| Layer | Tech | Why |
|-------|------|-----|
| Frontend | Angular 18 (standalone) + Material, RxJS | Enterprise SPA, streaming via Fetch |
| API | FastAPI + Uvicorn | Async, Pydantic validation, auto OpenAPI |
| PDF | PyMuPDF (`fitz`) | Fastest text extraction |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` (384-d) | Local, small, strong quality/size |
| Vector DB | ChromaDB (PersistentClient, cosine, HNSW) | Local, persistent, zero servers |
| LLM | Ollama + `llama3.1:8b` | 100% local, no API key |
| Validation | Pydantic v2 / pydantic-settings | Typed config + DTOs |
| HTTP client | httpx | Ollama health probe + timeouts |

**Config values (defensible in interview):** chunk_size=500, overlap=100, top_k=5, dim=384, temperature=0.1, num_ctx=8192, timeout=120s, max_question=2000, max_upload=25 MB.

---

## 3. END-TO-END FLOW (request)

```
User types question → chat.component.send()
  → chat.service.ask()/askStream()  (POST /api/ask or /api/ask/stream)
  → routes.ask()  → rag.answer()
      1. sanitize_question()            (helpers)      → clean/validate (400 if bad)
      2. retriever.retrieve()           (retriever)
           2a. embed_text(question)     (embedding)    → 384-d vector
           2b. vectordb.search(top_k=5) (vectordb)     → nearest chunks + distance
           2c. threshold + sort         (retriever)    → RetrievedChunk[]
      3. build_prompt()                 (prompt_builder) → system+user messages
      4. llm.generate()/stream()        (llm)          → Ollama answer
      5. RagAnswer{answer, sources, timings}           → JSON to Angular
  → chat.component renders answer + citations + timings
```

**Short-circuit:** if retrieval empty → return `"I don't know based on the provided documents."` **without** calling the LLM.

---

## 4. EXECUTION ORDER (startup)

```
uvicorn app.main:app
 → create_app() builds FastAPI, adds CORS, includes router, registers exception handlers
 → lifespan startup:
     ensure_directories() → embedding.warm_up() (load model)
     → llm.warm_up() (best-effort) → ingest_directory() (auto-ingest PDFs)
 → server ready on :8001
```

---

## 5. EVERY FILE IN 5 LINES

**config.py** — Pydantic `Settings` singleton (`@lru_cache`); env>.env>defaults; paths anchored to `__file__`; holds all knobs; `ensure_directories()`.

**utils/logger.py** — Configures root logging **once** (`@lru_cache`) with console + rotating file; `get_logger()`; `timer()` context manager measures retrieval/LLM ms.

**utils/helpers.py** — `clean_text`, `sanitize_question` (validate/len-check), `slugify`, `make_chunk_id` (deterministic → idempotent upsert), `file_sha1` (streamed dedupe hash).

**services/pdf_reader.py** — Stage 1: PyMuPDF opens PDF; per-page text; handles not-found/corrupt/encrypted/empty; returns `PdfDocument`; `read_pdf_bytes` for uploads.

**services/chunker.py** — Stage 2: manual **recursive** char splitter (¶→line→sentence→word→char), 500/100, attaches page+section metadata; returns `Chunk[]`.

**services/embedding.py** — Stage 3: lazy-loads MiniLM (singleton, thread-lock); `embed_texts` (batched, **L2-normalized**) → 384-d; `cosine_similarity` helper.

**services/vectordb.py** — Stage 4: `VectorStore` repository over ChromaDB (cosine/HNSW); `upsert_chunks`, `search`, `get`, `delete`, `count`, `reset`; `SearchResult.score = 1-distance`.

**services/retriever.py** — Stage 5: `Retriever.retrieve` embeds question → `vectordb.search` → threshold-filter+sort → `RetrievalResult` with `context_text()` + `sources()`.

**services/prompt_builder.py** — Stage 6: fixed grounded `SYSTEM_PROMPT`; fences context (anti-injection); budgets context to num_ctx; returns `BuiltPrompt` (system+user messages).

**services/llm.py** — Stage 7: Ollama client (singleton); `health`/`ensure_ready`; `generate` (blocking) + `stream` (tokens); typed errors (unavailable/missing/timeout).

**services/rag.py** — Orchestrator: `RagService.answer`/`answer_stream`; sanitize→retrieve→prompt→generate; timing breakdown; empty-retrieval short-circuit; `RagAnswer`.

**services/ingest.py** — Write orchestrator: read→chunk→embed→upsert; JSON **manifest** dedupe by file hash; `ingest_pdf_file/bytes/directory`; `force` re-ingest.

**api/models.py** — Pydantic DTOs: `AskRequest/AskResponse`, `SourceModel`, `TimingModel`, `IngestResponse`, `HealthResponse`, `ErrorResponse` (API contract + OpenAPI).

**api/routes.py** — Endpoints under `/api`: `ask`, `ask/stream`, `ingest`, `ingest/directory`, `documents`, `chunks`, delete, reset, health; maps domain errors → HTTP codes.

**main.py** — App factory; CORS; router; global exception handlers; lifespan warm-up + auto-ingest; `uvicorn app.main:app`.

**Frontend** — `chat.models.ts` (types mirror API), `chat.service.ts` (HttpClient + fetch NDJSON stream), `chat.component.ts` (chat UI, streaming, upload, health dot), `environment.ts` (apiBaseUrl).

---

## 6. IMPORTANT CLASSES & METHODS

- **Settings** (`get_settings()`), **PdfDocument/PdfPage**, **Chunk.metadata()**
- **VectorStore**: `upsert_chunks()`, `search()`, `delete_document()`, `count()`; **SearchResult.score**
- **Retriever.retrieve()**, **RetrievalResult.context_text()/sources()**, **RetrievedChunk.citation()**
- **build_prompt()**, **BuiltPrompt.to_messages()**, **SYSTEM_PROMPT**, **NO_ANSWER**
- **llm.generate()/stream()/health()**, **LLMResult**
- **RagService.answer()/answer_stream()**, **RagAnswer.as_dict()**, **RagTimings**
- **ingest_pdf_bytes()/ingest_directory()**, **IngestResult**

---

## 7. TOP INTERVIEW QUESTIONS (implementation)

**RAG / architecture**
1. Walk me through your ingest vs query paths.
2. How do you prevent hallucination? (grounded prompt + refusal + empty short-circuit)
3. How do citations work end-to-end? (page metadata → SearchResult → sources[])
4. Why manual pipeline instead of LangChain?
5. Where does streaming happen and how? (llm.stream → NDJSON → fetch reader)

**Chunking**
6. Why 500 chars / 100 overlap?
7. What is recursive character splitting and why prefer natural boundaries?
8. Why overlap at all? (facts on chunk boundaries)
9. How do you attach page/section metadata?
10. What happens to a single atom bigger than chunk_size? (recurse finer)

**Embeddings**
11. Why all-MiniLM-L6-v2? Why 384 dims?
12. Why L2-normalize embeddings? (dot product = cosine)
13. Why batch encode? Why a singleton model?
14. What thread-safety issue did you solve? (`_model_lock` on first load)
15. CPU vs GPU impact on latency?

**Vector DB**
16. Why ChromaDB? Why PersistentClient?
17. What distance metric and index? (cosine, HNSW)
18. How is upsert idempotent? (deterministic IDs)
19. How do you convert distance → score? (`1 - distance`, clamp)
20. How do you filter by document? (`where={"document_name":...}`)

**Retrieval**
21. What's top_k and why 5?
22. What is the min-score threshold for?
23. Why re-sort if Chroma returns nearest-first? (defensive)
24. How do you build the LLM context block? (numbered cited sources)
25. What if the collection is empty?

**Prompt**
26. What's in your system prompt? (grounding + refusal + anti-injection + cite)
27. How do you defend against prompt injection in retrieved text? (fenced, "untrusted data")
28. How do you budget context to the context window? (num_ctx × 4 × 0.6)
29. Why temperature 0.1?
30. Why the exact "I don't know" phrase?

**LLM / Ollama**
31. Why Ollama + llama3.1:8b?
32. How do you health-check the model? (`/api/tags`)
33. How do you handle timeouts / model-missing? (typed errors → 503/504)
34. Blocking vs streaming generate — implementation?
35. What is warm-up and why?

**FastAPI / API**
36. Why FastAPI over Flask?
37. How do domain errors map to HTTP codes?
38. How does file upload ingest work? (UploadFile → read_pdf_bytes)
39. How is NDJSON streaming returned? (StreamingResponse generator)
40. How is CORS configured and why?

**Config / ops**
41. How is config loaded & overridden? (Pydantic BaseSettings)
42. How do you guarantee a single settings instance? (`@lru_cache`)
43. How do you measure per-stage latency? (`timer()` → timings)
44. How do you avoid duplicate log handlers?
45. How does startup auto-ingest work? (lifespan → ingest_directory)

**Angular**
46. How does the frontend stream tokens? (fetch + ReadableStream reader, NDJSON parse)
47. Why fetch instead of HttpClient for streaming?
48. How do you show citations & timings?
49. How does the health dot work?
50. How is the API base URL configured per env?

---

## 8. TOP CROSS-QUESTIONS (the follow-ups)

1. ChromaDB → **How at 100M vectors?** (managed ANN: Azure AI Search / Milvus / pgvector; sharding)
2. top_k=5 → **What if the answer spans 10 chunks?** (raise top_k, re-rank, or larger chunks)
3. Overlap 100 → **Storage/duplication cost at scale?** (more chunks = more vectors; tune)
4. MiniLM → **When would 384 dims hurt?** (nuanced domain → bigger model / domain fine-tune)
5. Normalize → **Prove dot product = cosine.** (unit vectors ⇒ A·B = cosθ)
6. Deterministic IDs → **PDF edited by one line?** (hash differs; most chunks same → orphan cleanup)
7. Cosine → **Why not Euclidean?** (magnitude-invariant; better for text embeddings)
8. Recursive chunking → **Tables/forms?** (layout-aware extraction, structured chunks)
9. Temperature 0.1 → **Why not 0?** (0 fully greedy; 0.1 keeps slight robustness, still factual)
10. Anti-injection → **What's still exploitable?** (indirect injection via crafted docs; needs output filtering)
11. Ollama → **Concurrency limits?** (single-node queueing; scale with replicas / vLLM)
12. FastAPI async → **Is your embedding/LLM call blocking the loop?** (CPU-bound; use threadpool/offload)
13. Streaming → **How do you cancel mid-stream?** (AbortController on client; server generator stops)
14. Manifest dedupe → **Race on concurrent ingest?** (needs locking/DB, not a JSON file)
15. Health → **degraded vs down semantics?** (reachable but model missing = degraded)

---

## 9. PRODUCTION IMPROVEMENTS

- **AuthN/Z** (OAuth2/JWT, RBAC per claim), rate limiting, audit logging (insurance compliance).
- **Async offloading** of CPU-bound embed/LLM to worker pool; connection pooling.
- **Managed vector store** + metadata index; hybrid (keyword + vector) search + re-ranker.
- **Blob storage** for PDFs; OCR fallback (Document Intelligence) for scanned pages.
- **Observability**: structured JSON logs + request IDs, metrics, tracing (OpenTelemetry).
- **Caching** (Redis) for embeddings and frequent Q&A.
- **Docker/Kubernetes** deploy; HPA; secrets in Key Vault.
- **Eval harness** (groundedness, citation accuracy, latency SLOs).

---

## 10. AZURE MIGRATION MAP

| Local | Azure |
|-------|-------|
| Ollama llama3.1:8b | **Azure OpenAI** (gpt-4o) deployment |
| all-MiniLM-L6-v2 | **Azure OpenAI text-embedding-3-small/large** |
| ChromaDB | **Azure AI Search** (vector + semantic + filters) |
| Local PDFs | **Azure Blob Storage** + Document Intelligence |
| .env | **App Config + Key Vault** |
| File logs | **Application Insights** |
| uvicorn on host | **Container Apps / AKS** |

Code changes are small: `llm.py` swaps to Azure OpenAI SDK; `embedding.py` calls the embeddings endpoint; `vectordb.py` becomes an AI Search client — interfaces (`generate`, `embed_texts`, `search`) stay the same → **swap implementations, not the pipeline** (that's the payoff of your layered design).

---

## 11. KEY COMPARISONS (one-liners)

**FAISS vs ChromaDB** — FAISS = raw, fast ANN library, no persistence/metadata out of the box; ChromaDB = batteries-included (persistence, metadata filters, CRUD) — better for a full app, FAISS better for max-perf custom pipelines.

**Ollama vs Azure OpenAI** — Ollama = local, free, private, slower on CPU, you manage it; Azure OpenAI = managed, faster, scalable, paid, data leaves the box (unless private) — choice is privacy/cost vs scale/quality.

**SentenceTransformers vs OpenAI embeddings** — ST = local/free/384-d, good enough, no network; OpenAI = higher quality (1536-d+), managed, costs per token + network dependency.

**FastAPI vs Flask** — FastAPI = async, Pydantic validation, auto OpenAPI, native streaming; Flask = sync, minimal, more manual. FastAPI wins for typed AI APIs.

**PyMuPDF vs pdfplumber** — PyMuPDF = fastest raw text, C-backed; pdfplumber = better for **tables/layout** but slower. Use PyMuPDF for bulk text, pdfplumber/Document Intelligence for structured forms.

**Chunking strategies** — fixed-size (simple, cuts words), recursive char (yours: natural boundaries + overlap), semantic/embedding-based (split on topic shifts), layout-based (per section/table). Trade precision vs cost.

**Vector search** — embed query → ANN nearest neighbours (HNSW graph) by cosine distance → top_k. Approximate = fast at scale; exact = slow but precise.

**Cosine similarity** — `(A·B)/(|A||B|)`, range [-1,1]; with normalized vectors it's just `A·B`. Chroma distance = `1 - cosine`; your `score = 1 - distance`.

**Metadata filtering** — Chroma `where={"document_name": x}` pre-filters candidates before/with ANN so retrieval is scoped (per-document, per-section) — enables access control and precision.

**Prompt engineering (yours)** — role priming + strict grounding ("only from CONTEXT") + refusal phrase + fenced untrusted context (anti-injection) + citation nudge + low temperature = factual, cited, hard-to-hijack answers.

---

## 12. 60-SECOND PROJECT PITCH

"I built a fully local, offline RAG system for insurance claim documents. PDFs are read with PyMuPDF, split by a manual recursive character chunker at 500/100 with page and section metadata, embedded locally with all-MiniLM-L6-v2 into 384-d normalized vectors, and stored in ChromaDB with cosine similarity. On a question, I embed it, retrieve the top-5 nearest chunks, build a strictly grounded prompt that fences the context to prevent injection, and generate the answer with llama3.1:8b via Ollama — returning cited sources and per-stage timings. The whole thing is layered so each stage is swappable: moving to Azure OpenAI + Azure AI Search means changing three implementation files, not the pipeline. It's served by FastAPI with streaming NDJSON and an Angular 18 Material chat UI."
