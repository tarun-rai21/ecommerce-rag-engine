# E-Commerce Semantic Search & RAG Pipeline

[![CI](https://github.com/tarun-rai21/ecommerce-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/tarun-rai21/ecommerce-rag-engine/actions/workflows/ci.yml)

**Live demo: [ecommerce-rag-search.streamlit.app](https://ecommerce-rag-search.streamlit.app)** · API: [docs](https://ecommerce-rag-backend.onrender.com/docs), [readiness](https://ecommerce-rag-backend.onrender.com/readyz)

> Free hosting: the backend sleeps after 15 minutes without traffic, so the first search after a quiet spell can take about a minute while it wakes. Searches after that take well under a second plus the LLM call. Try "grey area rug 8x10" or "mid-century coffee table under $300".

A production-grade hybrid search and Retrieval-Augmented Generation (RAG) engine for e-commerce product discovery. Built with FastAPI, Qdrant, FastEmbed, Groq LLM, Redis, and Streamlit — fully containerized with Docker Compose.

---

## System Architecture & Motivation

### Why Naive RAG Fails in Production

Most RAG tutorials wire a single dense vector search to an LLM and call it done. This breaks in e-commerce for two reasons:

1. **SKU mismatch** — Dense embeddings capture semantic similarity, not exact token matches. A query for "Sony WH-1000XM5" may semantically retrieve "Bose QuietComfort 45" because they are conceptually similar, but the user wanted a specific SKU. Pure semantic search fails precision-critical queries.

2. **Keyword blindness** — Sparse retrieval (BM25/SPLADE) solves SKU matching but fails on abstract queries like "comfortable headphones for long flights" where no keyword overlap exists.

This system solves both by running dense and sparse retrieval in parallel and fusing results using Reciprocal Rank Fusion (RRF) — balancing semantic understanding with keyword precision.

**Measured result:** on held-out queries from the human-labelled Wayfair WANDS benchmark (42,994 products), hybrid retrieval scores **nDCG@10 = 0.726 vs 0.675 for dense-only** (+0.051, 95% CI [+0.034, +0.067]). See [Evaluation Results](#evaluation-results).

### Dataset: Wayfair WANDS
The catalog is [WANDS](https://github.com/wayfair/WANDS) (Wayfair ANnotation DataSet, MIT licence): **42,994 furniture and home products, 480 real search queries and 233,448 human relevance judgments** (Exact / Partial / Irrelevant). Having judgments written by someone other than the system's author is what makes the evaluation meaningful.

> **Prices are synthetic.** WANDS publishes no prices. Each product gets a deterministic synthetic price (log-normal around a per-category median, seeded by product id) so price filtering can be demonstrated. Every record is flagged `price_is_synthetic: true`, and the UI says so.

### Architecture Diagram
```mermaid
flowchart TB
    subgraph OFFLINE["⚙️ PART 1: DATA INGESTION (Offline Pipeline)"]
        direction LR
        A[("☁️ Raw Product\nData\nCSV / JSON")]
        B["🔍 Data Cleaning\n& Validation\nPydantic V2 Schemas"]
        C["🧠 Vector Embedding\nDense + Sparse\nFastEmbed · CPU Only"]
        D[("🗄️ Qdrant\nVector DB\nDocker · Persistent")]
        A --> B --> C --> D
    end

    subgraph ONLINE["🌐 PART 2: LIVE SEARCH & INFERENCE (Online Serving)"]
        direction TB
        U(["👤 User Query"])

        subgraph FASTAPI["⚡ FastAPI Backend — Async Core"]
            direction TB
            MW["🔀 Request Middleware\nLatency Timer · X-Process-Time"]
            CACHE[("⚡ Redis Cache\nFull Response · Fail-Open\nTTL: 5 min")]
            subgraph CORE["Core Orchestration Logic"]
                direction LR
                HS["🔎 Hybrid Search\nDense + Sparse\nParallel Execution"]
                RRF["📊 Reciprocal Rank\nFusion · k=30\nScore Merging"]
                HS --> RRF
            end
            LLM["🤖 LLM RAG Service\nStrict Prompting\nAnti-Hallucination Guard"]
            MW --> CACHE
            CACHE -->|"Cache Miss"| CORE
            CORE --> LLM
        end

        subgraph UI["🖥️ Streamlit UI · Port 8501"]
            direction TB
            SI["🔍 Search Input"]
            AD["💬 Answer Display\nStyled Markdown Block"]
            ED["📈 Engineering\nDiagnostics Sidebar\nLatency · RRF Scores"]
        end

        EXT["☁️ Groq LLM API\nopenai/gpt-oss-20b\nExponential Backoff"]

        U --> MW
        CACHE -->|"Cache Hit ⚡"| UI
        LLM -->|"Calls"| EXT
        EXT -->|"Answer"| LLM
        LLM -->|"Structured JSON"| UI
        SI & AD & ED
    end

    subgraph EVAL["📊 MONITORING & EVALUATION"]
        direction LR
        LOG["📝 Loguru\nStructured Logging\nPer-Request Metrics"]
        EVL["🧪 Eval Suite\nHit · P · Recall · MRR · nDCG\nKeyword Coverage · Gate"]
    end

    D -->|"Hybrid Query\nvia gRPC"| CORE
    OFFLINE -.->|"One-time Ingestion"| ONLINE
    ONLINE -.-> EVAL

    style OFFLINE fill:#1e293b,stroke:#475569,color:#e2e8f0
    style ONLINE fill:#1e3a5f,stroke:#3b82f6,color:#e2e8f0
    style FASTAPI fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    style CORE fill:#1e3a5f,stroke:#818cf8,color:#e2e8f0
    style UI fill:#1e3a5f,stroke:#34d399,color:#e2e8f0
    style EVAL fill:#1e293b,stroke:#475569,color:#e2e8f0
```

---

## Core Engineering Decisions

### 1. FastAPI Async/Await
FastAPI with async endpoints allows the server to handle multiple concurrent requests without blocking. During LLM API calls (200-800ms latency), the event loop serves other requests rather than sitting idle. Critical for production throughput under concurrent user load.

### 2. Pydantic V2 Validation
All incoming product records pass through strict Pydantic V2 schemas before touching the vector database. Negative prices, missing fields, and type mismatches are caught and logged at the ingestion boundary — not silently corrupting the index. Per-record error isolation means one bad row never crashes the pipeline.

### 3. Reciprocal Rank Fusion (RRF)
RRF merges dense and sparse retrieval rankings using the formula:

$$
RRF\_Score(d) = \sum_m \frac{1}{k + r_m(d)}
$$

Where `k` is a smoothing constant and `r_m(d)` is the 1-based rank of document `d` in retrieval model `m`. Fusion runs server-side in Qdrant (`RrfQuery(rrf=Rrf(k=30))` over two `Prefetch` legs of 100 candidates each). A client-side implementation with the same `k` is available via `USE_NATIVE_FUSION=false`, and a test asserts both paths rank identically.

Fusing ranks instead of scores avoids calibrating cosine similarity against BM25 scores, which live on different scales. `k` sets the trade-off between *agreement* and *top rank*: with a small `k` (Qdrant's default is 2) a document ranked 1st by one retriever ties with one ranked 4th by both; with a large `k` the document both retrievers agree on wins.

**Tuning (dev split only).** A 4×3 grid over `k ∈ {10, 30, 60, 100}` and candidates per leg `∈ {20, 50, 100}` was scored on the dev queries; `k=30`, 100 candidates had the best nDCG@10. Nearly all of the gain came from the candidate count (20 → 50: about +0.015 nDCG@10 at every `k`); differences between `k` values were within noise (< 0.005). On the untouched test split, the tuned setting scored 0.728 vs 0.701 for the common default (`k=60`, 20 candidates), both measured before the latest index rebuild.

> **BM25 needs Qdrant's IDF modifier.** FastEmbed's `Qdrant/bm25` model stores only the saturated term-frequency part of BM25. The IDF part is computed by Qdrant from collection statistics, and only when the sparse index is created with `modifier=Modifier.IDF`. Queries are encoded with `query_embed` (weight 1.0 per token). Ingestion refuses to reuse a collection whose sparse index lacks the modifier.

### 4. Response Caching (Redis)
- Redis with a 5-minute TTL caches the **full `/ask` response** (answer + retrieved products), so a cache hit returns exactly what the original request did
- Keys: `rag:<cache version>:ask:limit=<n>:<md5(normalized query)>`. Normalization (lowercase, trimmed, whitespace collapsed) makes "Waterproof  Bag" and "waterproof bag" share an entry; the version prefix is bumped on re-index so stale answers are never served
- Only successful LLM answers are cached, never fallback messages
- **Fail-open with a circuit breaker:** each Redis call has a 250 ms budget. A connection error bypasses the cache for 30 s, so an outage costs one failed call rather than one per request. Timeouts bypass it only after 5 in a row, because under load a single late call usually means a busy event loop, not a failed Redis (see the [load test](#load-test))
- Measured under load: a cache hit takes about 2 ms in the server and a median 52 ms at the client (most of it Docker Desktop's port forwarding on Windows), against a median 751 ms for a miss, which calls the LLM

### 5. Query Understanding & Hard Filters
Shoppers write constraints into the query: "grey area rug under $150", "desk lamp 4.5+ stars". `query_parser.py` turns them into structured filters and removes them from the text used for retrieval:

```
"grey area rug under $150"                    → text "grey area rug",     max_price 150
"coffee table between $200 and $400"          → text "coffee table",      min_price 200, max_price 400
"desk lamp 4.5+ stars"                        → text "desk lamp",         min_rating 4.5
```

- **Rules, not an LLM.** Price and rating constraints are numeric patterns, so rules are instant, free, deterministic and unit-testable, with no dependency on a rate-limited API. They're strict because a false positive silently hides good products. A number is a price only with a currency marker or directly after a price phrase, and never when followed by a unit ("48 inch vanity", "5 drawer dresser"). Positional words don't trigger ("twin **over** full bunk beds", "end table **between** recliners"). A test asserts that **none of the 480 real WANDS queries** produces a constraint.
- **Categories are not guessed from text.** A wrong category filter hides good results, and ranking already handles category meaning. Categories come from explicit facet filters (UI controls), populated from Qdrant's facet API (`GET /api/v1/facets`).
- **Pre-filtering, not post-filtering.** The Qdrant filter is applied inside each prefetch leg, so dense and BM25 each return their best *matching* candidates before fusion. Filtering after fusion would leave gaps (see the benchmark below).
- **Explicit filters override parsed ones** field by field, and the response reports `interpreted_query` and `applied_filters` so the UI can show what was searched. Filters are part of the cache key. The LLM is told which filters were applied, so it doesn't refuse with "I can't confirm the price".
- Products without a rating never match a rating filter: an unknown rating isn't evidence of a good one.

**Sizes are a preference, not a filter.** "grey area rug 8x10" searches for "grey area rug", then moves products sold in that size to the front of the query's 15 best matches. A hard filter would hide most suitable rugs, because only about 40% of rugs list a size. Catalog sizes rarely read the way shoppers write them, so they're rounded to the nearest foot: a 7' 10" × 10' 2" rug counts as an 8x10. Only room-scale pairs are read as sizes (both numbers at most 15 ft), and the response reports `preferred_size`.

### 6. CPU-Only Embedding (FastEmbed)
FastEmbed runs ONNX-optimized models locally with zero GPU requirement:
- Dense: `BAAI/bge-small-en-v1.5` (384-dim, ~67MB)
- Sparse: `Qdrant/bm25` (statistical, near-zero cost)

Both models are loaded once at startup as app-level singletons — not per-request — eliminating the 1000ms+ cold-start penalty on repeated queries.

**Indexing cost and the embedding cache.** On a 4-core laptop CPU (Intel i5-8265U), the dense model embeds about 10 products/s, so the first full index of WANDS takes about an hour. Embeddings are cached on disk in SQLite, keyed by `(model, sha256(text))`, so an interrupted run resumes where it stopped and re-indexing unchanged products runs at about 430 products/s. The embedded text is compact by design: name, product class, category path, a few attributes (color, material, style, shape) and the first 300 characters of the description. The full description stays in the payload for the LLM.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | Qdrant (Docker) |
| Dense Embeddings | BAAI/bge-small-en-v1.5 via FastEmbed |
| Sparse Embeddings | BM25 via FastEmbed |
| Fusion Algorithm | Reciprocal Rank Fusion (RRF, k=30, 100 candidates per leg) |
| Dataset | Wayfair WANDS (42,994 products, 480 labelled queries) |
| Backend API | FastAPI (async) + Uvicorn |
| LLM | Groq `openai/gpt-oss-20b` (configurable via `LLM_MODEL`) |
| Cache | Redis 7.4 (Docker) |
| Frontend | Streamlit |
| Validation | Pydantic V2 |
| Containerization | Docker Compose |

---

## Local Setup & Launch

### Prerequisites
- Docker Desktop
- Python 3.11+
- Groq API key (free at https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/tarun-rai21/ecommerce-rag-engine.git
cd ecommerce-rag-engine
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Launch the full stack
```bash
docker-compose up --build -d
```

This spins up 4 containers: Qdrant (v1.18.2), Redis (7.4.9), FastAPI backend, Streamlit frontend. Qdrant and Redis data live in named Docker volumes (`qdrant_data`, `redis_data`).

**Images.** Each service installs only its own dependencies (`requirements-backend.txt`, `requirements-frontend.txt`; `requirements.txt` includes both for local work). Both images run as a non-root user and use multi-stage builds with no compiler. The backend image bakes in both embedding models (65 MB) and then sets `HF_HUB_OFFLINE=1`, so a container starts without downloading anything and works on a host with no access to Hugging Face. Locally, the backend turns healthy about 13 s after starting, against roughly 30–50 s when it downloaded the models at startup.

| Image | Before | After |
|---|---|---|
| Backend (models included) | 1.02 GB, plus a 65 MB download at every start | 674 MB |
| Frontend | 1.02 GB | 806 MB, mostly Streamlit's own pandas and pyarrow dependencies |

Each container has a memory limit: Qdrant and the backend 1 GB, the frontend 512 MB, Redis 256 MB. Redis evicts the least-recently-used answers at 200 MB instead of being killed. In use, the stack takes about 0.6 GB. On Windows, Docker Desktop's VM takes up to half of RAM by default; a `[wsl2] memory=4GB` line in `%UserProfile%\.wslconfig` caps it.

### 4. Download and ingest product data
```bash
pip install -r requirements.txt
python scripts/download_wands.py          # ~96 MB into data/wands/ (not committed)
python -m src.ingestion.ingest            # WANDS → collection "wands_products"
```
Options: `--dataset sample` (the original 39-product demo), `--collection <name>`, `--limit N`, `--recreate` (rebuild from scratch), `--no-cache`.

Ingestion validates every record, creates payload indexes on `category`, `product_class`, `price`, `rating` and `review_count`, and writes a JSON report to `reports/` (counts, cache reuse, rejection reasons). It refuses to reuse an existing collection whose sparse index lacks the IDF modifier (`--recreate` rebuilds it). The first full run takes about an hour on a laptop CPU; later runs reuse the embedding cache.

### Run the tests
```bash
pip install -r requirements-dev.txt
pytest
```
The suite uses Qdrant's local mode and the real embedding models, so it does not need Docker. Lint with `ruff check .`.

**CI** (`.github/workflows/ci.yml`) runs on every push and on pull requests to `master`: ruff, the full test suite on Python 3.11 (with the embedding models and pip packages cached between runs), and a build of both Docker images. It needs no secrets, because the tests replace the LLM with fakes.

### 5. Access the application
- **Frontend UI**: http://localhost:8501
- **Backend API docs**: http://localhost:8000/docs
- **Qdrant dashboard**: http://localhost:6333/dashboard

---

## Observability

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
# Grafana http://localhost:3000 (anonymous read-only)  ·  Prometheus http://localhost:9090
```

**Metrics.** `GET /metrics` serves Prometheus metrics. With `METRICS_TOKEN` set, as on Render, it needs `Authorization: Bearer <token>`. It isn't listed in the API docs.

| Metric | What it shows |
|---|---|
| `rag_requests_total{route,method,status}` | Traffic and error rate per route |
| `rag_request_duration_seconds{route}` | Latency histogram per route |
| `rag_stage_duration_seconds{stage}` | Where `/ask` time goes: `cache_lookup`, `search` (includes query embedding), `llm` |
| `rag_answers_total{source}` | Answers from `llm`, `cache`, `no_results`, or `llm_error` (the shopper got the fallback) |
| `rag_llm_tokens_total{type}` | Billed prompt and completion tokens, which give spend at the provider's price |
| `rag_cache_lookups_total{result}` | `hit`, `miss`, `error`, `bypassed` (breaker open) |
| `rag_cache_breaker_open` | 1 while Redis is being skipped |
| `rag_rate_limited_total{scope}`, `rag_auth_failures_total` | Requests the safeguards refused |

Labels only ever hold route templates and fixed values, never query text or ids, so the number of series stays bounded. A scanner probing random paths adds to `route="unmatched"`, not one series per path. Every known label value is created at 0 on startup. Otherwise a series would first appear with a non-zero value, and `rate()` would miss its first events.

**Dashboard.** `observability/grafana/dashboards/rag-engine.json` is generated by `observability/build_dashboard.py` and provisioned into Grafana with the Prometheus data source. It has 13 panels:
- headline stats: requests/s, 5xx rate, p95 `/ask` latency, cache hit ratio, estimated LLM cost per hour, breaker state;
- traffic and latency by route;
- p95 latency by stage;
- answers by source;
- tokens per minute;
- cache outcomes;
- refused requests.

Each query was checked against live traffic. For example, after a run of searches and questions:
- cache hit ratio 43%;
- p95 for the LLM stage 2.2 s, search 0.1 s and cache lookup 4.5 ms;
- 895 prompt and 143 completion tokens per minute, about $0.007 per hour.

**Logs.** `LOG_FORMAT=json` writes one JSON object per line for a log platform to index, which Render uses. Every line written while handling a request carries its `request_id`, the same id returned in `X-Request-ID` and in error bodies, so a support report can be traced to its log lines. Tracebacks don't print local variables (`diagnose=False`), because those can hold keys or queries, and the Redis URL with its password is never logged.

On the free deployment, Prometheus and Grafana run locally only. Pointing a hosted Prometheus or Grafana Cloud at the backend's `/metrics` with the token is the next step.

## Deployment

### Live demo: free managed services

Running at [ecommerce-rag-search.streamlit.app](https://ecommerce-rag-search.streamlit.app), with the backend at [ecommerce-rag-backend.onrender.com](https://ecommerce-rag-backend.onrender.com/readyz).

```
Visitor ──▶ Streamlit Community Cloud (UI)
               │  HTTPS + X-API-Key, sent from Streamlit's server (the browser never sees the key)
               ▼
            Render (FastAPI backend, Docker image from Dockerfile.backend)
               ├──▶ Qdrant Cloud   (index, free 1 GB cluster)
               ├──▶ Upstash Redis  (answer cache over TLS, rediss://)
               └──▶ Groq           (LLM)
```

Each service runs on its own free tier. The backend is configured by environment only: `QDRANT_URL` and `QDRANT_API_KEY` replace the local host and ports, `REDIS_URL` replaces the Redis host and port, and `render.yaml` holds the Render service as code, with the secrets entered in the dashboard.

- **Render's free plan** gives 512 MB of RAM and 0.1 CPU, and sleeps after 15 minutes without traffic. Under those limits, simulated locally with `docker run --cpus 0.1 --memory 512m`, the backend:
  - uses 292 MB;
  - takes 0.17–0.42 s per search, against about 0.03 s on a 4-core laptop;
  - needs about 60 s from a cold start to ready, mostly loading the models.

  With `EMBED_THREADS=1`, ONNX Runtime runs one thread per model instead of competing for the small CPU share. The UI waits up to 90 s and tells the visitor that the first search can be slow while the backend wakes.
- **Loading the index:** `python deploy/upload_snapshot.py` sends a snapshot of the local collection (195 MB) to Qdrant Cloud. It asks for the cluster URL and API key and doesn't print or store them.
- **Streamlit Community Cloud** installs `src/frontend/requirements.txt`, which a test keeps identical to `requirements-frontend.txt`. It reads `BACKEND_URL` and `API_KEY` from the app's secrets, which it also exposes as environment variables.

### AWS (planned)

The same Docker stack runs on a single EC2 instance, with [Caddy](https://caddyserver.com) in front for HTTPS:

```
Internet ──HTTPS──▶ Caddy :80/:443 (automatic Let's Encrypt certificate, HTTP → HTTPS)
                      ├── /                → Streamlit UI
                      └── /api, /docs, ... → FastAPI (API key required on /api/v1)
                                              ├── Qdrant  (not exposed; loopback only for loading the index)
                                              ├── Redis   (not exposed)
                                              └── Groq API
```

- **Instance:** t3.small (2 vCPU, 2 GB RAM) in ap-south-1 with Ubuntu 24.04, plus 2 GB of swap. The stack uses about 0.6 GB in steady state.
- **Address:** an Elastic IP, with a free `sslip.io` hostname (`13-232-1-2.sslip.io` resolves to `13.232.1.2`) so Let's Encrypt can issue a certificate without a purchased domain.
- **Firewall:** the security group opens 80 and 443 to everyone and 22 (SSH) to one IP only. Only Caddy publishes ports.
- **Cost:** about $20/month on demand: the instance about $16, the public IPv4 address $3.60 and a 20 GB gp3 disk about $2.
- **The index isn't rebuilt on the server.** Embedding 43k products takes about an hour on a laptop and far longer on a t3.small, so `deploy/upload-index.sh` sends a Qdrant snapshot of the local collection (195 MB) over SSH and restores it in about 15 s.
- **Single-instance limits:** the server isn't redundant, and rate limits and the cache are per process. The whole UI shares one rate-limit bucket on `/ask`, because the Streamlit server makes the calls. That protects the Groq quota, but at busy moments visitors can be asked to retry.

**Steps:**

```bash
# 1. One-time server setup (Docker, swap, automatic security updates, clone)
ssh -i key.pem ubuntu@<ip> 'bash -s' < deploy/setup-server.sh

# 2. On the server: ~/ecommerce-rag-engine/.env with
#    DOMAIN=<ip-with-dashes>.sslip.io, GROQ_API_KEY=..., API_KEYS=<random>, FRONTEND_API_KEY=<same>
#    (generate a key with: openssl rand -hex 24)

# 3. Start Qdrant, then load the index from the local machine
ssh -i key.pem ubuntu@<ip> 'cd ecommerce-rag-engine && docker compose -f deploy/docker-compose.prod.yml --env-file .env up -d qdrant'
deploy/upload-index.sh ubuntu@<ip> key.pem

# 4. Start everything (builds the images on the server)
ssh -i key.pem ubuntu@<ip> 'cd ecommerce-rag-engine && docker compose -f deploy/docker-compose.prod.yml --env-file .env up -d --build'

# Update after a push: git pull, then step 4 again. Smoke test: https://<domain>/readyz
```

The whole production stack was first rehearsed locally with `DOMAIN=localhost`. The snapshot restored all 42,994 points with the BM25 IDF modifier intact, HTTP redirected to HTTPS, `/api/v1` refused requests without a key, `/ask` answered through Caddy, the Streamlit UI loaded over its WebSocket, and nothing but Caddy answered on the host's ports.

## API Reference

### Safeguards

Settings come from the environment (see `.env.example`). The defaults suit local development: no key, no limits. A public deployment sets all of them.

| Setting | Effect |
|---|---|
| `API_KEYS` | Comma-separated keys. When set, every `/api/v1` route needs one in the `X-API-Key` header, otherwise `401`. Keys are compared in constant time. |
| `RATE_LIMIT_PER_MINUTE` | Per client on `/search` and `/facets`. The client is the API key when one is sent, else the IP. Over the limit: `429` with `Retry-After`. |
| `ASK_RATE_LIMIT_PER_MINUTE` | The same for `/ask`, separately, because `/ask` can call the paid LLM. |
| `CORS_ORIGINS` | Browser origins allowed to call the API; empty means none. The Streamlit UI calls the API from its server, so it needs none. |
| `DOCS_ENABLED` | Serves `/docs` and the OpenAPI schema. |

- **Rate limits** use a token bucket per client, so short bursts up to the per-minute figure are allowed. The buckets are in memory, so each process keeps its own count; several instances would need a shared store such as Redis.
- **Questions** have control and invisible formatting characters (null bytes, zero-width spaces, text-direction overrides) replaced by spaces before validation.
- **Qdrant calls** time out after `QDRANT_TIMEOUT_SECONDS` (5 s).
- **The UI's key** (`FRONTEND_API_KEY`) is sent from the Streamlit server, so it never reaches the browser.

**Errors** all share one format, and the rejected input is never echoed back:

```json
{"error": {"code": "rate_limited", "message": "Too many requests: the limit is 5 per minute. Try again in 12 s.",
           "request_id": "49479d8553b744669219c1b2fa63292a"}}
```

Codes: `invalid_request` (422, with a `details` list of fields), `unauthorized` (401), `rate_limited` (429), `not_found` (404), `search_unavailable` (503, Qdrant unreachable) and `internal_error` (500). Every response carries an `X-Request-ID` header: the caller's own, if it's short and plain, otherwise a new one. The same id is in the server log line for that request.

### GET /healthz and GET /readyz

Probes outside `/api/v1` that need no key. `/healthz` answers whenever the process is up. `/readyz` returns `200` only if Qdrant answers and the collection has products, and reports Redis and the LLM without depending on them, because the API still works without its cache or its answers:

```json
{"status": "ready", "checks": {"qdrant": {"ok": true, "points": 42994}, "cache": {"ok": true}, "llm": {"ok": true}}}
```

Docker Compose uses `/readyz` as the backend's health check, and the frontend starts once the backend is healthy.

### POST /api/v1/search
Hybrid semantic + keyword search. Returns ranked products with RRF scores.

```json
Request:  {"q": "grey area rug under $150", "limit": 5,
           "filters": {"categories": ["Rugs"], "min_rating": 4},   // optional, overrides parsed values
           "parse_filters": true}                                  // optional, default true
Response: {"results": [...], "latency_ms": 45.2,
           "interpreted_query": "grey area rug",
           "applied_filters": {"min_price": null, "max_price": 150.0, "categories": ["Rugs"], "min_rating": 4.0}}
```

### GET /api/v1/facets?key=category
Distinct values with counts for filter dropdowns (`key`: `category` or `product_class`).

```json
Response: {"key": "category", "values": [{"value": "Furniture", "count": 16039}, ...]}
```

### POST /api/v1/ask
Full RAG pipeline. Returns LLM-generated answer with diagnostic metadata.

```json
Request:  {"q": "mid-century modern coffee table", "limit": 5}     // limit: 1–20
Response: {
  "answer": "Here are mid-century modern coffee tables from the catalog...",
  "source": "llm",          // llm | cache | no_results | llm_error
  "retrieved_products": [...],
  "timing": {
    "cache_lookup_ms": 1.2,
    "search_ms": 22.4,
    "llm_ms": 210.5,
    "total_ms": 234.1
  },
  "interpreted_query": "mid-century modern coffee table",
  "applied_filters": {...},
  "preferred_size": null,   // e.g. "8x10" when the query names a room size
  "usage": {"prompt_tokens": 1026, "completion_tokens": 176}  // of the LLM call; a cache hit costs nothing
}
```
`/ask` accepts the same `filters` and `parse_filters` fields as `/search`.

---

## Evaluation Results

### Retrieval on WANDS (primary benchmark)

```bash
python -m src.evaluation.retrieval_eval --split dev    # tuning
python -m src.evaluation.retrieval_eval --split test   # reporting (run once per frozen config)
```

The script runs the production `ProductSearchService` in-process (no HTTP, no cache, no LLM) in three modes (dense only, BM25 only, hybrid) and scores each against the WANDS human judgments. Queries are split by a hash of the query id into **dev (179 queries, tuning)** and **test (301 queries, reporting)**. Confidence intervals are 95% bootstrap intervals over queries (2,000 resamples); comparisons against dense are paired (same queries, same resamples).

**Test split, 301 queries, frozen config (k=30, 100 candidates per leg):**

| Mode | nDCG@10 | Recall@100 | MRR@10 | P@10 | judged@10 | Search p50 / p95 |
|---|---|---|---|---|---|---|
| Dense only | 0.675 [0.641, 0.709] | 0.624 [0.578, 0.665] | 0.658 [0.606, 0.708] | 0.454 [0.406, 0.502] | 0.79 | 13 / 16 ms |
| BM25 only | 0.659 [0.622, 0.696] | 0.659 [0.612, 0.701] | 0.723 [0.672, 0.771] | 0.460 [0.414, 0.506] | 0.79 | 9 / 12 ms |
| **Hybrid (RRF)** | **0.726 [0.693, 0.758]** | **0.675 [0.629, 0.715]** | **0.740 [0.691, 0.786]** | **0.493 [0.444, 0.540]** | 0.85 | 14 / 19 ms |

Paired difference vs dense: **hybrid nDCG@10 +0.051 [+0.034, +0.067]**, Recall@100 +0.051 [+0.026, +0.076], both significant. BM25 alone vs dense: nDCG@10 −0.017 [−0.044, +0.009] (not significant). For reference, hybrid with the untuned default (`k=60`, 20 candidates) scored nDCG@10 0.701 on test (measured before the tie-breaking fix and the latest index rebuild).

**Reproducibility.** RRF often produces exact score ties (a product ranked *r*-th by dense alone scores the same as one ranked *r*-th by BM25 alone), and Qdrant returns tied points in an order that varies between runs. That shifted nDCG@10 by up to ±0.002 between identical runs. Both fusion paths and the single-retriever baselines now break ties by product id (fetching a few extra results so a tie at the cut-off can't change the set). Two consecutive test runs now produce identical top-10s for every query in every mode. Rebuilding the index from scratch can still move scores by a few thousandths (hybrid went from 0.728 to 0.726), so differences that small aren't meaningful. Latency columns vary with machine load and are indicative only.

**How the metrics are defined:**
- **nDCG@10** (primary) uses graded gains: Exact = 2, Partial = 1.
- **Recall@100, MRR@10 and P@10** treat only *Exact* as relevant, so they use the 237 test queries with at least one Exact judgment.
- **Unjudged products count as irrelevant**, the standard convention for pooled judgments. **judged@10** is the share of the top 10 that has any judgment: around 80% here, so absolute scores are somewhat pessimistic for every mode.
- **Duplicate judgments:** 1,467 (query, product) pairs are judged more than once and 14 disagree; the highest label is kept.
- **Top-10 vs depth-100:** top-10 metrics come from a `limit=10` search with the production settings; Recall@100 comes from a separate `limit=100` search.
- **Search latency** excludes query embedding and is measured on a laptop with Qdrant in Docker.

### Price-constrained retrieval (filters)

```bash
python -m src.evaluation.filter_eval --split test
```

WANDS queries never state a budget, so the benchmark adds one. For each test query with at least one Exact judgment, the budget **X = median synthetic price of its Exact products** (about half of them fit), and the query becomes "*query* under $X". A product counts as relevant only if it's judged relevant **and** priced ≤ X. All three systems use the same hybrid retrieval:

| System (237 test queries) | nDCG@10 (budget-aware) | Over-budget results | Slots filled (of 10) |
|---|---|---|---|
| No parsing: search the raw text, no filter (previous behaviour) | 0.297 [0.268, 0.328] | 50.4% | 10.0 |
| Post-filter: parse, search, then drop over-budget results | 0.391 [0.355, 0.426] | 0.0% | 5.0 |
| **Pre-filter: parse, filter inside each prefetch (this system)** | **0.551 [0.503, 0.597]** | **0.0%** | **10.0** |

Pre-filtering vs no parsing: **+0.254 nDCG@10** [+0.224, +0.284]; vs post-filtering: **+0.160** [+0.140, +0.184]. Without parsing, half of what's shown is over budget. Post-filtering never violates the budget but leaves half the page empty. Prices are synthetic, so this measures filter mechanics, not real-world price behaviour.

### Room-size rug queries

```bash
python -m src.evaluation.size_eval
```

40 rug queries with a room size, top 5 results, 95% bootstrap CIs. WANDS has no size judgments, so this measures whether results list the size and still match the rest of the query, not human-judged relevance.

| System | Results listing the size | Queries with at least one | Results mentioning the requested term | Results that are rugs |
|---|---|---|---|---|
| Size left in the search text | 15.0% [11.0%, 19.0%] | 65.0% [50.0%, 80.0%] | 99.0% [97.5%, 100%] | 95.5% [91.5%, 99.0%] |
| **Size as a preference (this system)** | **53.5% [44.0%, 63.5%]** | **97.5% [92.5%, 100%]** | **98.5% [96.5%, 100%]** | **96.0% [91.5%, 99.0%]** |

Results listing the size rose by 38.5 points [+30.0, +47.0], and results mentioning the requested term changed by −0.5 points [−2.0, +1.0], which isn't significant. The re-ordering depth of 15 was chosen on these same 40 queries, so the numbers are somewhat optimistic.

**A design that was rejected.** Adding sizes to the search text first looked like a clear win: results listing the size rose from 15.5% to 92.5%, with nDCG@10 down only 0.002, which passed the rule set before the experiment. A further check showed the cost that metric hid: results mentioning the requested term fell from 98.5% to 82.5%, so "grey area rug 8x10" returned espresso and green rugs. Re-ordering all 100 candidates failed in a similar way (−8.5 points), which led to the depth of 15.

### Answer quality

```bash
python -m src.evaluation.generation_eval
python -m src.evaluation.generation_eval --calibrate src/evaluation/judge_labels.csv --report <labelled report>.json

# While trying a change: run only the affected groups and compare with the last full run
python -m src.evaluation.generation_eval --segments sized ooc --compare reports/<full run>.json
```

`--segments` (`plain`, `constrained`, `ooc`, `sized`) runs part of the sample. The report is saved as `generation-subset_*`, so it's never mistaken for a full run. `--compare` prints paired changes on the shared items: refusals that flipped, and answers that gained or lost full support, with a CI. A full run is still needed before reporting numbers.

110 queries go through the production path (parsing, hybrid search, context, generator): 50 WANDS dev queries, 20 more with a budget appended, 30 hand-written queries for products a home-goods catalog doesn't stock (each checked against the catalog's top results), and 10 rug queries with a room size. An LLM judge from a different model family (`qwen/qwen3.8-27b`) audits each answer against its context, and rule-based checks cover refusals, prices and wording. 95% bootstrap CIs.

| Metric | Result |
|---|---|
| Correct refusal, out-of-catalog | 100% (30/30) |
| Refusal on in-catalog queries | 14.3% [7.1%, 22.9%] (10/70) |
| ...when an Exact product was among the 5 retrieved | 3.7% [0.0%, 9.3%] (2/54) |
| Refusal on rug queries with a room size | 0% (0/10) |
| Claims the judge found supported | 99.0% [98.3%, 99.6%] (1,687 of 1,701) |
| Answers with every claim supported (judge) | 84.3% [75.7%, 92.9%]; see calibration below |
| Relevance (judge, 1-5) | 4.79 [4.69, 4.87] |
| Answers quoting a price not in the context | 0% (0/70) |
| Answers where a whole-catalog claim was removed in code | 14.3% (10/70) |

**Room-size queries took four rounds.** The final version differs from the first in four ways:

1. **Softer refusal rule.** The model refuses only when no product is the kind of item requested, and may show a product whose requested detail isn't listed. "grey area rug 8x10" was refused before. This let one out-of-catalog query through ("men's running shoes size 10" returned shoe racks), so the rule now adds that a merely related product, such as storage or an accessory, isn't the item asked for. That only partly works: the evaluation's one answer to the shoe query was a refusal, but repeating the identical request 5 times gave 1 refusal and 4 shoe-rack answers.
2. **Sizes in the context.** The model sees each rug's sizes, but it still named only 22 of the 31 retrieved rugs that fit. All 9 it left out had near sizes, such as 7' 10" × 10' 2" for an 8x10. A rule in the system prompt didn't help (21/31).
3. **A line naming the rugs that fit.** The context now opens by listing the products sold at or close to the requested size, with their sizes. The model then named 30 of 31 (a rough title match on the answers), and "grey area rug 8x10" shows all three grey rugs that fit instead of one. A side effect: answers began calling the list "the only ones in our catalog".
4. **Whole-catalog claims removed in code.** The prompt already forbade those claims and a warning in the context didn't stop them, so `clean_output` now deletes any sentence that claims the products cover the whole catalog, before the answer is shown or judged. It removed one from 10 of 70 answers here. It works on wording, so a paraphrase can still slip through: the judge flagged one ("these are the ones that match").

Against the version before these changes, on the same queries: out-of-catalog refusals 29 → 30 of 30 (but see the shoe query above), in-catalog refusals unchanged at 10 of 70, and answers with every claim supported 62 → 59 of 70, a change of −4.3 points [−15.7, +7.1] that isn't significant. On the 10 rug queries, 6 answers are fully supported against 9 before. Answers there now list 3 to 5 rugs each, and the unsupported claims are mostly pile heights stated for the wrong rug. These steps were tuned on the same 10 rug queries they're measured on, so treat the rug numbers as development results.

**Prompt iteration.** The first version told shoppers "these are the only products in our catalog" in 25% of answers (15/59), although the model sees 5 products. Two prompt revisions brought this to 1.7%, a paired change of −22 points [−36, −10], with no significant change in supported claims (−0.2 points [−1.0, +0.8]). Answers grew about 20% longer. The same queries were used to develop the prompt and report these results, and the WANDS queries all come from the dev split, so treat them as development numbers rather than a held-out estimate.

**Judge calibration: the judge is lenient.** A second AI reviewer (Claude Opus 5.5, not a human) labelled 30 answers against their context, with the sheet in `src/evaluation/judge_labels.csv`. These answers came from the previous 100-query run, before the sizes change:

| | |
|---|---|
| Every claim supported? | 77% agreement, Cohen's κ 0.46 |
| Rated fully supported | Reviewer 60%, judge 83% |
| Relevance | 87% exact, 100% within 1 point |
| Disagreements on support | 7, all where the reviewer found a problem the judge passed |

Read the 84.3% above as an upper estimate. The misses are mostly small embellishments ("classic design", a product name read as a material). Limits: the reviewer is an AI, it had already seen the judge's flagged claims for this run, so agreement may be inflated, it marked an answer down for a single unsupported word, and 30 answers is a small sample. A human review is still needed before calling the judge human-validated.

Every LLM call is cached by its full request in `.cache/llm_calls.sqlite`, so an unchanged re-run is free. A cold run takes about 2 hours on Groq's free tier, and longer when its rate limits force long waits.

### Load test

```bash
python -m benchmarks.load_test                    # about 15 minutes; needs the stack running
python -m benchmarks.load_test --parts search     # /search only, no LLM calls
```

Closed-loop users send requests back to back with no think time: 5 s warm-up, then 30 s measured per level, using WANDS test queries. The client runs on the same 4-core laptop as the stack (i5-8265U, Docker capped at 4 GB), so it competes for CPU and the numbers understate what the server alone could do. Every `/ask` cache miss calls Groq, whose free tier allows only a few calls a minute, so `/ask` is measured in three parts. Latency in ms with 95% bootstrap CIs.

**`/search`** (parsing, embedding, Qdrant hybrid search, RRF; no cache):

| Users | Requests/s | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1 | 14.4 | 75 [73, 76] | 89 [87, 92] | 104 [95, 121] | 0 |
| 4 | 37.4 | 103 [102, 104] | 147 [143, 153] | 182 [175, 204] | 0 |
| 8 | 37.3 | 206 [203, 209] | 313 [302, 323] | 405 [387, 423] | 0 |
| 16 | 42.0 | 375 [371, 378] | 500 [489, 520] | 586 [574, 625] | 0 |
| 32 | 34.9 | 855 [843, 872] | 1,468 [1,357, 1,559] | 1,882 [1,723, 2,081] | 0 |

Throughput levels off at about 40 requests/s from 4 users, where the CPU-bound query embedding saturates the four cores; beyond that, extra users only queue.

**`/ask`, cache miss** (20 questions, one at a time, 20 s apart): median 751 ms end to end, of which the LLM call is 704 ms, search 26 ms and the cache lookup 1 ms. A prompt is about 1,000 tokens (855–1,174), and an answer has a median of 176 completion tokens (29–380, including reasoning).

**`/ask`, cache hit** (the same 20 questions, after the fix below):

| Users | Requests/s | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1 | 18.7 | 52 [52, 52] | 64 [62, 65] | 69 [67, 72] | 0 |
| 8 | 124.4 | 59 [59, 60] | 98 [95, 100] | 154 [140, 170] | 0 |
| 16 | 122.5 | 116 [115, 117] | 207 [202, 212] | 409 [364, 417] | 0 |
| 32 | 79.1 | 134 [131, 138] | 1,545 [1,452, 1,667] | 2,590 [2,387, 2,957] | 0 |

The server spends about 2 ms on a hit; most of the ~50 ms floor is Docker Desktop's port forwarding on Windows.

**A failure the load test found.** In the first run at 32 users, one Redis lookup overran its 250 ms budget because the event loop was busy. The circuit breaker treated that as Redis being down and skipped the cache for 30 s, so every request went to search and the LLM, and Groq rejected 53 calls for rate limits. Shoppers saw "please try again later", p95 reached 9.7 s, and throughput fell to 21 requests/s. The breaker now opens at once on a connection error, as before, but only after 5 timeouts in a row; any success resets the count. In the rerun, 32 users got 79 requests/s, p95 1.5 s and no errors. No lookup timed out in the rerun, so the new rule wasn't exercised there; unit tests cover it.

### Cost per query

```bash
python -m benchmarks.cost      # from the token usage recorded by the load test
```

Only an uncached `/ask` costs money: embedding and search run on our own CPU, and a cache hit makes no LLM call. Prices are Groq's on-demand rates for `openai/gpt-oss-20b`, read on 2026-11-09 from the [models page](https://console.groq.com/docs/models): $0.075 per 1M input tokens and $0.30 per 1M output tokens. Token counts come from the load test's 20 uncached answers, and completion tokens include the model's reasoning.

| | Mean | Range |
|---|---|---|
| Prompt tokens per answer | 1,031 | 855–1,174 |
| Completion tokens per answer | 176 | 29–380 |
| Cost per LLM answer | $0.00013 | $0.00008–$0.00020 |

| Cache hit rate | Cost per 1,000 questions |
|---|---|
| 0% | $0.130 [$0.114, $0.146] |
| 25% | $0.098 [$0.086, $0.110] |
| 50% | $0.065 [$0.057, $0.073] |
| 75% | $0.033 [$0.029, $0.037] |

The hit rate depends on real traffic, which this project doesn't have, so cost is shown for several rates. Input is 59% of the cost. The context is capped for this reason: product descriptions are cut to 400 characters. Two discounts would lower these figures but aren't counted: Groq bills cached prompt tokens at half price, which could apply to the shared system prompt, and its batch API is half price but suits offline work only. The evaluation judge (`qwen/qwen3.8-27b`) costs $0.80 per 1M input and $4.00 per 1M output tokens, about 11–13 times the generator per token, but it runs only during evaluation.

### Cross-encoder reranking: tested, not enabled

A cross-encoder reranker (`src/backend/services/rerank.py`) can re-score the top 20 hybrid results by reading the query and product text together. It runs under a time budget and falls back to the RRF order on timeout or error. It's **off by default** (`RERANK_ENABLED=false`) because, on the dev split, no CPU-feasible model cleared the bar set before the experiment: a gain over plain hybrid that is statistically significant *and* at least +0.010 nDCG@10.

| Model (top 20 re-scored) | nDCG@10 vs hybrid (dev, 179 queries) | Added latency p50 / p95 |
|---|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` | +0.012 [−0.003, +0.028] | +0.78 s / +1.1 s |
| `jinaai/jina-reranker-v1-tiny-en` | +0.013 [−0.003, +0.029] | +0.64 s / +0.9 s |
| `jinaai/jina-reranker-v1-turbo-en` | +0.013 [−0.002, +0.030] | +0.99 s / +1.5 s |

The gain is consistent across models but too small to separate from noise on 179 queries, and it would multiply search latency by about 40× on this CPU. `BAAI/bge-reranker-base` (3.1 pairs/s here, about 6.5 s for 20 candidates) was ruled out on speed before any accuracy run. These rerankers were trained on web passages (MS MARCO), not product listings, which may limit transfer. A domain-tuned reranker or GPU serving would change the trade-off; to try it, set `RERANK_ENABLED=true` and `RERANK_MODEL`, and evaluate with `--modes hybrid hybrid_rerank --baseline hybrid`. The test split wasn't used for any reranker decision.

### Smoke evaluation (original 39-product demo catalog)

```bash
python -m src.ingestion.ingest --dataset sample --collection ecommerce_products
# start the backend with COLLECTION_NAME=ecommerce_products, then:
python -m src.evaluation.eval                 # retrieval + answers over HTTP
python -m src.evaluation.eval --skip-answers   # retrieval only (no LLM calls)
```

This end-to-end check calls the running API. Retrieval metrics come from `/search` (no cache, no LLM); answer metrics come from `/ask`. Each run writes a JSON report to `reports/`. The results below come from the Phase 0 configuration (`k=60`, 20 candidates per leg).

#### Latest Results (39 products, 15 test queries, k=5)

| Metric | Score | Notes |
|---|---|---|
| Hit Rate @ 5 | 1.0000 | |
| Precision @ 5 | 0.2533 | Ceiling on this ground truth is 0.2933 (only 22 relevant labels across 75 slots) |
| Recall @ 5 | 0.9222 | |
| MRR @ 5 | 1.0000 | A relevant product is ranked first for every query |
| nDCG @ 5 | 0.9429 | Binary relevance |
| Keyword coverage | 0.8667 | Share of expected keywords in the answer (`openai/gpt-oss-20b`); varies slightly between runs |

Retrieval metrics were identical across repeated runs, including runs served entirely from cache.

**How to read these numbers honestly:**
- **This set is saturated.** Dense-only retrieval also reaches Hit@5 = 1.0, so these 15 queries can't show whether hybrid retrieval beats a single retriever; that's what the WANDS benchmark above is for.
- **Keyword coverage is not faithfulness.** It checks whether expected words appear in the answer, not whether the answer is supported by the retrieved products.

#### Evaluation Gate
The script exits with code `1` if Hit Rate@5 falls below `0.70` or any retrieval request fails. It's meant to be run in CI; no CI workflow is set up yet.

---

## Project Structure
```bash
ecommerce-rag-engine/

├── config/

│   └── settings.py           # Pydantic-settings config management

├── data/

│   ├── raw/

│   │   └── sample_products.json   # 39-product demo catalog

│   └── wands/                # WANDS CSVs (downloaded, not committed)

├── scripts/

│   └── download_wands.py

├── src/

│   ├── ingestion/

│   │   ├── loaders/          # sample + WANDS loaders

│   │   ├── schemas.py        # Pydantic V2 product validation + embedded text

│   │   ├── pricing.py        # Deterministic synthetic prices (disclosed)

│   │   ├── embedder.py       # FastEmbed dense + sparse

│   │   ├── embedding_cache.py  # SQLite cache keyed by text hash (resumable)

│   │   └── ingest.py         # Batch ingestion pipeline + payload indexes + report

│   ├── backend/

│   │   ├── main.py           # FastAPI app + lifespan
│   │   ├── metrics.py        # Prometheus metrics + /metrics
│   │   ├── logging_config.py # Text or JSON logs with request ids

│   │   ├── api/
│   │   │   ├── health.py     # /healthz, /readyz
│   │   │   ├── errors.py     # Error envelope + exception handlers
│   │   │   ├── security.py   # API keys + token-bucket rate limiting
│   │   │   └── middleware.py # Request id, access log, latency header

│   │   ├── api/v1/

│   │   │   ├── endpoints.py  # /search and /ask routes

│   │   │   └── router.py

│   │   ├── models/

│   │   │   └── api_schemas.py

│   │   └── services/

│   │       ├── search.py     # Hybrid search + RRF + payload filters + facets

│   │       ├── query_parser.py  # "under $300", "4+ stars" → SearchFilters

│   │       ├── rerank.py     # Optional cross-encoder reranker (off by default)

│   │       ├── cache.py      # Redis cache service

│   │       └── llm.py        # Groq LLM service

│   ├── frontend/

│   │   ├── app.py            # Streamlit UI

│   │   └── components/

│   │       └── diagnostics.py

│   └── evaluation/

│       ├── datasets.py       # WANDS judgments + fixed dev/test split

│       ├── metrics.py        # Hit@K, P@K, Recall@K, MRR, nDCG, keyword coverage

│       ├── stats.py          # Bootstrap CIs + paired differences

│       ├── retrieval_eval.py # WANDS benchmark: dense vs BM25 vs hybrid

│       ├── filter_eval.py    # Price-constrained benchmark: no parsing vs post- vs pre-filter
│       ├── size_eval.py      # Room-size benchmark: size as search text vs preference
│       ├── generation_eval.py # Answer quality: refusals, faithfulness, relevance, calibration
│       ├── judge.py          # LLM judge, rule-based answer checks, cached LLM calls
│       ├── out_of_catalog.json # Queries the catalog can't answer
│       ├── judge_labels.csv  # Reviewer labels used to calibrate the judge

│       ├── ground_truth.json # 15 queries for the demo catalog

│       └── eval.py           # HTTP smoke evaluation + gate

├── deploy/
│   ├── docker-compose.prod.yml  # EC2 stack: Caddy + backend + frontend + Qdrant + Redis
│   ├── Caddyfile             # HTTPS + routing (/ → UI, /api → backend)
│   ├── setup-server.sh       # One-time Ubuntu setup: Docker, swap, updates, clone
│   ├── upload-index.sh       # Copy the local Qdrant collection to the server as a snapshot
│   └── upload_snapshot.py    # Copy it to Qdrant Cloud instead (prompts for URL + key)
├── observability/
│   ├── prometheus.yml        # Scrape config for the backend's /metrics
│   ├── build_dashboard.py    # Generates the Grafana dashboard JSON (dashboard as code)
│   └── grafana/              # Provisioned data source + dashboard
├── docker-compose.observability.yml  # Optional Prometheus + Grafana
├── render.yaml               # Render Blueprint for the backend (free plan)
├── benchmarks/
│   ├── load_test.py          # Closed-loop load test: /search, /ask misses and hits
│   └── cost.py               # Cost per 1,000 questions from recorded token usage
├── tests/                    # pytest: ingestion, search, API, cache, metrics, LLM

├── Dockerfile.backend

├── Dockerfile.frontend

├── docker-compose.yml

├── requirements.txt

└── requirements-dev.txt
```

---

## Known Limitations & Future Work

- **Synthetic prices**: WANDS has no prices; the ones shown are generated and must not be read as real.
- **Answer quality has no human validation yet**: the judge was calibrated against an AI reviewer that wasn't fully blind, on 30 answers. It also misses subtle embellishments, so its faithfulness scores are upper estimates.
- **No reranker**: cross-encoder reranking was evaluated and left off (gain not significant on dev, about +0.7–1.0 s per query on CPU); see above.
- **One answer per query**: the evaluation stores a single answer per query, and the generator runs at temperature 0.2, so the same request can be refused one time and answered the next. Refusal rates carry more uncertainty than their intervals show. "men's running shoes size 10" is refused only about 1 time in 5, with shoe racks offered otherwise.
- **Whole-catalog claims are removed by pattern**: the model still writes "these are the only ones in our catalog" in about 1 answer in 7. Code removes the sentences it recognises, but a paraphrase can get through.
- **Rug answers make more unsupported claims**: answers to room-size queries list more rugs, and 4 of 10 had a claim the judge couldn't support, mostly a pile height stated for the wrong rug.
- **Only about 40% of rugs list a size**, so size is a ranking preference rather than a guarantee.
- **Constraint parsing covers price, rating and rug room sizes only**, in English patterns. Categories aren't inferred from text (by design), and other attributes ("walnut") are left to ranking rather than enforced as filters.
- **Incomplete judgments**: about 20% of top-10 results have no WANDS judgment and count as irrelevant, so absolute scores are pessimistic.
- **Embedding model**: `bge-small-en-v1.5` is optimized for speed. `bge-large-en-v1.5` or `text-embedding-3-large` would improve semantic retrieval quality at higher compute cost.
- **Sparse model**: BM25 is statistical. Upgrading to SPLADE neural sparse model would improve keyword expansion for ambiguous queries.

