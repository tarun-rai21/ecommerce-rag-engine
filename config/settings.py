from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Qdrant ────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    COLLECTION_NAME: str = "wands_products"

    # ── Retrieval ─────────────────────────────────────────
    # Tuned on the WANDS dev split (2026-11-01): k=30, prefetch=100 had the best nDCG@10.
    # Most of the gain came from prefetch 20 → 50+; k in {10..100} differed by < 0.005.
    RRF_K: int = Field(default=30, ge=1)
    PREFETCH_LIMIT: int = Field(default=100, ge=1)
    # True: fuse server-side with Qdrant's RRF. False: two queries + Python RRF.
    # Both paths use RRF_K, so they must produce the same ranking.
    USE_NATIVE_FUSION: bool = True

    # ── Reranking ─────────────────────────────────────────
    # Off: on the WANDS dev split no CPU-feasible model gave a significant gain over
    # hybrid (+0.012–0.013 nDCG@10, CIs include 0) for +0.7–1.0 s/query. See README.
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = Field(default=20, ge=2)
    RERANK_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0)

    # ── LLM ───────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    # llama-3.1-8b-instant was retired by Groq (404 model_not_found, 2026-11-01).
    # gpt-oss-20b handled a 15-request burst without 429s; qwen/qwen3.8-27b is faster
    # but its free tier caps output at 1,000 tokens/min.
    LLM_MODEL: str = "openai/gpt-oss-20b"
    # Passed as `reasoning_effort` when set (gpt-oss models); empty string = omit.
    LLM_REASONING_EFFORT: str = "low"
    LLM_TIMEOUT_SECONDS: float = 20.0

    # ── Cache ─────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6379)
    CACHE_TTL_SECONDS: int = Field(default=300, ge=1)
    CACHE_OP_TIMEOUT_SECONDS: float = Field(default=0.25, gt=0)
    CACHE_COOLDOWN_SECONDS: float = Field(default=30.0, ge=0)
    CACHE_BREAKER_TIMEOUTS: int = Field(default=5, ge=1)  # timeouts in a row before Redis is skipped
    # Bump whenever the index or response format changes so stale entries are never served.
    CACHE_VERSION: str = "v3"  # v3: WANDS index + filters in the response

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()
