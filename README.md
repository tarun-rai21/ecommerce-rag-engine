# E-Commerce Semantic Search & RAG Pipeline

A production-grade hybrid search and Retrieval-Augmented Generation (RAG) engine for e-commerce product discovery. Built with FastAPI, Qdrant, FastEmbed, Groq LLM, Redis, and Streamlit — fully containerized with Docker Compose.

---

## System Architecture & Motivation

### Why Naive RAG Fails in Production

Most RAG tutorials wire a single dense vector search to an LLM and call it done. This breaks in e-commerce for two reasons:

1. **SKU mismatch** — Dense embeddings capture semantic similarity, not exact token matches. A query for "Sony WH-1000XM5" may semantically retrieve "Bose QuietComfort 45" because they are conceptually similar, but the user wanted a specific SKU. Pure semantic search fails precision-critical queries.

2. **Keyword blindness** — Sparse retrieval (BM25/SPLADE) solves SKU matching but fails on abstract queries like "comfortable headphones for long flights" where no keyword overlap exists.

This system solves both by running dense and sparse retrieval in parallel and fusing results using Reciprocal Rank Fusion (RRF) — balancing semantic understanding with keyword precision.

**Measured result:** on held-out queries from the human-labelled Wayfair WANDS benchmark (42,994 products), hybrid retrieval scores **nDCG@10 = 0.728 vs 0.676 for dense-only** (+0.052, 95% CI [+0.035, +0.069]). See [Evaluation Results](#evaluation-results).

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

**Tuning (dev split only).** A 4×3 grid over `k ∈ {10, 30, 60, 100}` and candidates per leg `∈ {20, 50, 100}` was scored on the dev queries; `k=30`, 100 candidates had the best nDCG@10. Nearly all of the gain came from the candidate count (20 → 50: about +0.015 nDCG@10 at every `k`); differences between `k` values were within noise (< 0.005). On the untouched test split, the tuned setting scores 0.728 vs 0.701 for the common default (`k=60`, 20 candidates).

> **BM25 needs Qdrant's IDF modifier.** FastEmbed's `Qdrant/bm25` model stores only the saturated term-frequency part of BM25. The IDF part is computed by Qdrant from collection statistics, and only when the sparse index is created with `modifier=Modifier.IDF`. Queries are encoded with `query_embed` (weight 1.0 per token). Ingestion refuses to reuse a collection whose sparse index lacks the modifier.

### 4. Response Caching (Redis)
- Redis with a 5-minute TTL caches the **full `/ask` response** (answer + retrieved products), so a cache hit returns exactly what the original request did
- Keys: `rag:<cache version>:ask:limit=<n>:<md5(normalized query)>`. Normalization (lowercase, trimmed, whitespace collapsed) makes "Waterproof  Bag" and "waterproof bag" share an entry; the version prefix is bumped on re-index so stale answers are never served
- Only successful LLM answers are cached, never fallback messages
- **Fail-open with a circuit breaker:** each Redis call has a 250 ms budget; after a failure the cache is bypassed for 30 s, so an outage costs one timeout rather than one per request
- Measured locally: cache hit ≈ 1.5–11 ms end-to-end vs ≈ 300–950 ms for a full search + LLM request

### 5. CPU-Only Embedding (FastEmbed)
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
The suite uses Qdrant's local mode and the real embedding models, so it does not need Docker.

### 5. Access the application
- **Frontend UI**: http://localhost:8501
- **Backend API docs**: http://localhost:8000/docs
- **Qdrant dashboard**: http://localhost:6333/dashboard

---

## API Reference

### POST /api/v1/search
Hybrid semantic + keyword search. Returns ranked products with RRF scores.

```json
Request:  {"q": "mid-century modern coffee table", "limit": 5}
Response: {"results": [...], "latency_ms": 45.2}
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
  }
}
```

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
| Dense only | 0.676 [0.642, 0.710] | 0.629 [0.583, 0.669] | 0.657 [0.605, 0.708] | 0.451 [0.404, 0.500] | 0.79 | 11 / 15 ms |
| BM25 only | 0.659 [0.622, 0.695] | 0.659 [0.612, 0.701] | 0.723 [0.672, 0.771] | 0.460 [0.414, 0.506] | 0.79 | 7 / 10 ms |
| **Hybrid (RRF)** | **0.728 [0.695, 0.760]** | **0.674 [0.628, 0.714]** | **0.744 [0.694, 0.790]** | **0.492 [0.443, 0.538]** | 0.85 | 12 / 15 ms |

Paired difference vs dense: **hybrid nDCG@10 +0.052 [+0.035, +0.069]**, Recall@100 +0.046 [+0.023, +0.070], both significant. BM25 alone vs dense: nDCG@10 −0.017 [−0.045, +0.008] (not significant). For reference, hybrid with the untuned default (`k=60`, 20 candidates) scores nDCG@10 0.701 on test.

**How the metrics are defined:**
- **nDCG@10** (primary) uses graded gains: Exact = 2, Partial = 1.
- **Recall@100, MRR@10 and P@10** treat only *Exact* as relevant, so they use the 237 test queries with at least one Exact judgment.
- **Unjudged products count as irrelevant**, the standard convention for pooled judgments. **judged@10** is the share of the top 10 that has any judgment: around 80% here, so absolute scores are somewhat pessimistic for every mode.
- **Duplicate judgments:** 1,467 (query, product) pairs are judged more than once and 14 disagree; the highest label is kept.
- **Top-10 vs depth-100:** top-10 metrics come from a `limit=10` search with the production settings; Recall@100 comes from a separate `limit=100` search.
- **Search latency** excludes query embedding and is measured on a laptop with Qdrant in Docker.

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

│   │   ├── api/v1/

│   │   │   ├── endpoints.py  # /search and /ask routes

│   │   │   └── router.py

│   │   ├── models/

│   │   │   └── api_schemas.py

│   │   └── services/

│   │       ├── search.py     # Hybrid search + RRF

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

│       ├── ground_truth.json # 15 queries for the demo catalog

│       └── eval.py           # HTTP smoke evaluation + gate

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
- **Answer quality isn't measured yet**: retrieval is benchmarked on WANDS, but answers aren't. An LLM-judge check of answers against the retrieved context is planned.
- **Over-strict refusals**: the prompt tells the model to refuse when the context lacks a requested detail. For example, "grey area rug 8x10" retrieves grey rugs, but size isn't among the attributes passed to the LLM, so it refuses.
- **No filters yet**: payload indexes exist, but query parsing into filters ("under $200", category) isn't implemented, so constraints aren't enforced by retrieval.
- **Incomplete judgments**: about 20% of top-10 results have no WANDS judgment and count as irrelevant, so absolute scores are pessimistic.
- **Embedding model**: `bge-small-en-v1.5` is optimized for speed. `bge-large-en-v1.5` or `text-embedding-3-large` would improve semantic retrieval quality at higher compute cost.
- **Sparse model**: BM25 is statistical. Upgrading to SPLADE neural sparse model would improve keyword expansion for ambiguous queries.

