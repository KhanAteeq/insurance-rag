# Insurance RAG — Azure Deployment & Interview Guide

A complete record of how this project was deployed to Azure end-to-end, plus an
interview-ready script. Use it to rehearse before interviews.

**Live setup:** Angular UI → **Azure App Service** (FastAPI) → **Azure OpenAI
gpt-5-mini** + **text-embedding-3-small** + ChromaDB. Feature-flagged so the same
code runs locally on Ollama + MiniLM (`BACKEND_MODE=local|azure`).

---

## PART 1 — Deployment, step by step

### A. Prepare the code (in the workspace)
1. **Feature flag** `BACKEND_MODE` (`local` | `azure`) added to `config.py` — one
   codebase, two backends.
2. **Azure adapters** added behind existing interfaces:
   - `llm.py` → Azure OpenAI **gpt-5-mini** via `chat.completions`
     (`reasoning_effort=minimal`, `max_completion_tokens` — gpt-5 is a reasoning model).
   - `embedding.py` → Azure **text-embedding-3-small** (1536-dim).
   - `vectordb.py` → separate ChromaDB collection for Azure (1536-d vs local 384-d).
3. **Slim cloud requirements** (`requirements-azure.txt`) — no PyTorch/
   sentence-transformers (~2 GB saved); `sentence_transformers` import made lazy.
4. **SQLite fix** — Azure's image ships an old sqlite3 that ChromaDB rejects. Added
   `pysqlite3-binary` + a swap in `app/__init__.py` that runs before ChromaDB imports.
5. **Lightweight startup** — on App Service, skip auto-ingest at boot (passes the
   Free-tier startup probe); ingest on demand via `POST /api/ingest/directory`.

### B. Azure resources & tooling
6. Installed **Azure CLI** (`winget install Microsoft.AzureCLI`); logged in with
   `az login --use-device-code` (device code was more resilient on a flaky network).
7. Tried **Azure Container Registry** → trial subscription **blocks ACR cloud build**,
   and no local Docker → pivoted away from containers.
8. Chose **Azure App Service** (Python 3.12, Linux, **Free F1**). Hit **VM quota = 0**
   → fixed by selecting a different **region** (West Central US).

### C. Configure & deploy the backend
9. **App settings (env vars):**
   ```
   BACKEND_MODE=azure
   AZURE_OPENAI_ENDPOINT=https://<foundry-resource>.services.ai.azure.com
   AZURE_OPENAI_API_KEY=<key>            # from the resource's Keys and Endpoint
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   AZURE_EMBEDDING_DIMENSION=1536
   SCM_DO_BUILD_DURING_DEPLOYMENT=true
   ```
10. **Startup command** (set via **Azure Cloud Shell**, which runs inside Azure and
    bypassed the network resets):
    ```bash
    az webapp config set -g rag-rg-group -n insurance-rag-api \
      --startup-file "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    ```
11. **Built a deploy zip** (app code + slim `requirements.txt` + PDFs; excluded
    `.env`, caches, local vector data).
12. **Deployed via Kudu ZipDeploy** (browser drag-drop at
    `https://<app>.scm.<region>.azurewebsites.net/ZipDeployUI`) — the CLI deploy kept
    getting reset by the network; the browser upload succeeded. Oryx then ran `pip install`.
13. **Restarted** (`az webapp restart` in Cloud Shell) → `/api/health` = **status: ok**.
14. **Ingested** the corpus: `POST /api/ingest/directory` → **12 files, 107 chunks**
    embedded via Azure and stored in ChromaDB.
15. **Verified RAG:** `POST /api/ask` → *"The bond number is SUR8843012… (Source 1)"*
    in ~2.6 s.

### D. Connect the frontend
16. Set `apiBaseUrl` in `frontend/src/environments/environment.ts` to the App Service
    URL. Backend CORS already allows `http://localhost:4200`, so the local UI calls the
    cloud backend directly.
17. Tested at `http://localhost:4200` → status **Ready**, asked a question → grounded,
    cited answer from Azure gpt-5-mini. ✅
    (For a fully-cloud frontend: build with `ng build` and deploy `dist/.../browser` to
    **Azure Static Web Apps**, then add the SWA URL to `CORS_ALLOW_ORIGINS`.)

### Problems solved (memorize these)
| Problem | Fix |
|--------|-----|
| Model "deprecated / Deploy disabled" | Deploy in a model-rich region (gpt-5-mini) |
| Azure OpenAI quota error | Lower capacity (TPM) / different deployment type |
| ACR cloud build blocked (trial) | App Service from source instead of containers |
| App Service VM quota = 0 | Different region |
| CLI `ConnectionReset` on every call | Use **Azure Cloud Shell** (runs in Azure) |
| Zip deploy reset | **Kudu ZipDeploy** browser drag-drop |
| Default "Hey Python" page | Set the **Startup Command** |
| ChromaDB SQLite crash | **pysqlite3-binary** swap in `app/__init__.py` |

### Teardown (stop the credit meter)
- Portal → `insurance-rag-api` → **Stop**, or
- Portal → Resource groups → `rag-rg-group` → **Delete**.

---

## PART 2 — Interview script

### 60-second overview
"I built a Retrieval-Augmented Generation system for insurance claim documents and
deployed it end-to-end on Azure. The backend is a FastAPI service on **Azure App
Service**, using **Azure OpenAI gpt-5-mini** for generation and
**text-embedding-3-small** for embeddings, with **ChromaDB** as the vector store. The
Angular frontend talks to it over REST with streaming responses.

The key design decision was **decoupling** — the LLM, embeddings, and vector store each
sit behind their own interface, controlled by a `BACKEND_MODE` flag. So the same
codebase runs fully offline on Ollama locally, or on Azure OpenAI in the cloud, by
changing configuration, not code."

### "What went wrong and how I solved it"
"Deploying on a trial subscription surfaced real production issues:
- The **container build was blocked** and I had no Docker, so I pivoted to **App Service
  deploying Python from source**.
- I hit **quota limits** on the model and compute, solved via **region** choice and
  capacity tuning.
- ChromaDB **crashed on Azure's old system SQLite**, fixed by injecting
  `pysqlite3-binary` before the import.
- I **slimmed the cloud build** by making PyTorch lazy (embeddings come from Azure),
  cutting ~2 GB.
- I made **startup lightweight** so it passes the Free-tier health probe."

### Architecture soundbite
"Because it's layered, moving from local to Azure — or from Azure OpenAI to a
self-hosted model — is a **configuration change, not a rewrite**."

### Follow-up Q&A
- **Why App Service not containers?** "The trial blocked cloud container builds and I had
  no Docker. App Service builds Python from source via Oryx. In a real org I'd
  containerize and use Container Apps or AKS."
- **How do you keep secrets safe?** "The key is an App Service application setting
  injected as an env var — never in code or the image. Next step: Key Vault references."
- **How would you make it production-grade?** "Azure AI Search for the vector store,
  auth + per-claim RBAC, an API gateway with rate limiting, Redis cache, and App Insights
  monitoring."
- **What did deployment teach you?** "Working locally is half the job — quotas, regions,
  base images, startup probes, and cost controls are where real deployments live. I
  learned to read platform logs and isolate each failure."
- **Cost?** "Near-zero on Free tiers; gpt-5-mini is cheap per token. In production I'd
  tier models, cache, and keep retrieved context small to control the LLM bill."

### Closer
"It's not just a notebook demo — it's a layered, cloud-deployed RAG service where I owned
the full path from code to a running Azure endpoint, and debugged real platform issues to
get there."
