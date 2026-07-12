# Azure Deployment Guide — Insurance RAG (Practice)

Deploy the app to Azure cheaply, then delete it. Beginner-friendly, Portal-based.
**Golden rules:** keep everything in ONE resource group; use Free tiers; delete at the end.

Target: **Azure OpenAI** (LLM + embeddings) · **Azure AI Search** (vectors) ·
**Container Apps** (backend) · **Static Web Apps** (frontend).

---

## PHASE 0 — Budget alert (do this first)
1. Portal → search **Cost Management** → **Budgets** → **+ Add**.
2. Amount **₹500**, alert at **80%** → add your email → Create.
> This just warns you; it won't stop services. Your $200 credit covers this practice easily.

---

## PHASE 1 — Resource group
1. Portal → search **Resource groups** → **+ Create**.
2. Name **`rg-insurance-rag`**, Region **Central India** → Review + create → Create.

---

## PHASE 2 — Azure OpenAI (LLM + embeddings)
### 2a. Create the resource in a model-rich region
Some regions only show **deprecated** models (Deploy button greyed out). Use a region that has current models.
1. Portal → search **Azure OpenAI** → **+ Create**.
2. RG **`rg-insurance-rag`**, **Region: East US 2** (or **Sweden Central**), Name **`oai-insurance-rag`**, Tier **Standard S0** → Create.

### 2b. Deploy the two models
1. Open the resource → **Go to Azure AI Foundry portal**.
2. **Deployments → + Deploy model → Deploy base model**.
3. Deploy a **current small chat model** whose **Deploy** button is enabled (e.g. `gpt-4o-mini`, `gpt-4.1-mini`, or `o4-mini` — whichever is current & enabled). **Write down the deployment name** (e.g. `gpt-4o-mini`).
4. Deploy **`text-embedding-3-small`** (1536-dim). Write down its deployment name.
> If Deploy is disabled for ALL chat models, your region is wrong — delete this resource and recreate it in **East US 2** or **Sweden Central**.

### 2c. Copy connection details
1. Back in the Azure portal on `oai-insurance-rag` → **Keys and Endpoint**.
2. Copy **Endpoint** (`https://oai-insurance-rag.openai.azure.com/`) and **Key 1**.
   Keep the key private (you'll paste it into Azure app settings later, not into any chat).

**You now have:** endpoint, key, chat deployment name, embedding deployment name.

---

## PHASE 3 — Azure AI Search (vectors, Free tier)
1. Portal → search **Azure AI Search** → **+ Create**.
2. RG **`rg-insurance-rag`**, Name **`srch-insurance-rag`**, Region **Central India**, **Pricing tier → Free** → Create.
3. Open it → copy the **Url** (`https://srch-insurance-rag.search.windows.net`).
4. **Settings → Keys** → copy the **Primary admin key** (keep private).
> The app will **create the index automatically** on first ingest (schema: id, text, metadata, 1536-dim vector). You don't need to build it by hand.
> ⚠️ Embeddings change from 384-dim (local MiniLM) to **1536-dim** (Azure), so you must **re-ingest** the PDFs once running on Azure — the old ChromaDB vectors are not reused.

---

## PHASE 4 — Code changes (I implement these in your workspace)
Your app is layered, so only three files + config change, behind a flag `BACKEND_MODE`:
- `llm.py` → Azure OpenAI chat (uses `openai.AzureOpenAI`).
- `embedding.py` → Azure OpenAI embeddings (`text-embedding-3-small`, 1536-dim).
- `vectordb.py` → Azure AI Search (index + vector query).
- `requirements.txt` → add `openai`, `azure-search-documents`.
- New env vars (see Phase 6).
`BACKEND_MODE=local` keeps your machine working with Ollama + ChromaDB; `BACKEND_MODE=azure` uses the cloud services.

---

## PHASE 5 — Build & push the backend image
You need the image in a registry. Two options:

### Option A — No Docker on your PC (Azure builds it)
1. Portal → search **Container registries** → **+ Create** → RG `rg-insurance-rag`, Name **`acrinsurancerag<random>`**, SKU **Basic** → Create.
2. Install Azure CLI locally (or use **Cloud Shell** in the portal, the `>_` icon).
3. From the `backend` folder run:
   ```
   az login
   az acr build --registry acrinsurancerag<random> --image insurance-rag-api:v1 .
   ```
   This uploads your code and builds the image in Azure using your `Dockerfile`.

### Option B — Docker Desktop installed
```
az acr login --name acrinsurancerag<random>
docker build -t acrinsurancerag<random>.azurecr.io/insurance-rag-api:v1 ./backend
docker push acrinsurancerag<random>.azurecr.io/insurance-rag-api:v1
```

---

## PHASE 6 — Deploy backend to Container Apps
1. Portal → search **Container Apps** → **+ Create**.
2. RG `rg-insurance-rag`, App name **`ca-insurance-rag`**, Region **Central India**, create a new **Environment** (Consumption).
3. **Container tab:** uncheck "Use quickstart image" → source **Azure Container Registry** → select your registry + image `insurance-rag-api:v1`. (Enable **Admin user** on the ACR under Access keys if prompted.)
4. **Ingress tab:** Enabled → **External** → **Target port 8000**.
5. **Create.** After deploy, open it → copy the **Application Url** (e.g. `https://ca-insurance-rag.<region>.azurecontainerapps.io`).
6. Open the app → **Settings → Containers → Environment variables → Edit and deploy** → add:
   ```
   BACKEND_MODE=azure
   AZURE_OPENAI_ENDPOINT=https://oai-insurance-rag.openai.azure.com/
   AZURE_OPENAI_API_KEY=<your key>
   AZURE_OPENAI_API_VERSION=2024-10-21
   AZURE_OPENAI_CHAT_DEPLOYMENT=<chat deployment name>
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   AZURE_SEARCH_ENDPOINT=https://srch-insurance-rag.search.windows.net
   AZURE_SEARCH_API_KEY=<search admin key>
   AZURE_SEARCH_INDEX=insurance-claims
   EMBEDDING_DIMENSION=1536
   CORS_ALLOW_ORIGINS=["https://<your-static-web-app-url>"]
   ```
   Save (it redeploys). Test: open `<app-url>/api/health` → expect `status: ok`.

---

## PHASE 7 — Deploy frontend to Static Web Apps (Free)
1. Push your `frontend` folder to a **GitHub repo**.
2. In `frontend/src/environments/environment.prod.ts` set `apiBaseUrl` to your Container App URL.
3. Portal → search **Static Web Apps** → **+ Create** → RG `rg-insurance-rag`, Name `swa-insurance-rag`, Plan **Free**, connect **GitHub** → pick repo/branch.
4. Build details: **Build Preset: Angular**, App location `/frontend` (or `/` if repo root is frontend), Output location **`dist/insurance-rag-frontend/browser`** → Create.
5. It auto-builds via GitHub Actions. Copy the Static Web App URL.
6. Back in Container Apps env var `CORS_ALLOW_ORIGINS`, make sure it contains this URL → redeploy.

---

## PHASE 8 — Ingest & test
1. Open the Static Web App URL → the chat UI loads; the health dot should be green.
2. Upload a PDF (or POST to `<app-url>/api/ingest/directory` if PDFs are baked in).
   > This creates the Azure AI Search index and embeds with Azure (1536-dim).
3. Ask a question → you should get a cited answer from **gpt-4o-mini**.

---

## PHASE 9 — TEARDOWN (stop all charges)
1. Portal → **Resource groups** → **`rg-insurance-rag`** → **Delete resource group**.
2. Type the name to confirm → Delete. Everything (OpenAI, Search, Container App, ACR, Storage) is removed.
3. (Optional) Delete the GitHub repo / Static Web App if created separately.

---

## Cost reminder
- Azure AI Search **Free** = ₹0. Static Web Apps **Free** = ₹0.
- Container Apps scale-to-zero = ~₹0 idle.
- Azure OpenAI = a few ₹ for test questions (gpt-4o-mini is cheap).
- ACR Basic ≈ ₹420/mo (delete after practice).
- **Total practice cost: well under ₹200**, covered by your $200 credit — as long as you delete the resource group when done.
