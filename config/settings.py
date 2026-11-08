from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Qdrant ────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    COLLECTION_NAME: str = "wands_products"
    QDRANT_TIMEOUT_SECONDS: int = Field(default=5, ge=1)
    # Managed Qdrant (e.g. Qdrant Cloud): when QDRANT_URL is set it replaces host/ports,
    # e.g. https://xyz.us-east-1-0.aws.cloud.qdrant.io:6333 (gRPC goes to 6334 over TLS).
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

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

    # ── Embedding ─────────────────────────────────────────
    # ONNX Runtime threads per model; 0 = its default (one per core). On a host with a
    # fraction of a CPU, 1 avoids threads fighting over it and saves memory.
    EMBED_THREADS: int = Field(default=0, ge=0)

    # ── Cache ─────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6379)
    # Managed Redis (e.g. Upstash): when set it replaces host/port. rediss:// = TLS.
    REDIS_URL: str = ""
    CACHE_TTL_SECONDS: int = Field(default=300, ge=1)
    CACHE_OP_TIMEOUT_SECONDS: float = Field(default=0.25, gt=0)
    CACHE_COOLDOWN_SECONDS: float = Field(default=30.0, ge=0)
    CACHE_BREAKER_TIMEOUTS: int = Field(default=5, ge=1)  # timeouts in a row before Redis is skipped
    # Bump whenever the index or response format changes so stale entries are never served.
    CACHE_VERSION: str = "v3"  # v3: WANDS index + filters in the response

    # ── API safeguards ────────────────────────────────────
    # Defaults suit local development (open, unlimited); a public deployment must set
    # API_KEYS and the rate limits.
    API_KEYS: str = ""  # comma-separated keys accepted in X-API-Key; empty = no auth
    RATE_LIMIT_PER_MINUTE: int = Field(default=0, ge=0)  # /search and /facets, per client; 0 = off
    ASK_RATE_LIMIT_PER_MINUTE: int = Field(default=0, ge=0)  # /ask calls the paid LLM, so set it lower
    CORS_ORIGINS: str = ""  # comma-separated browser origins allowed to call the API; empty = none
    DOCS_ENABLED: bool = True  # /docs, /redoc and /openapi.json

    def qdrant_client_kwargs(self) -> dict:
        """Connection arguments for QdrantClient / AsyncQdrantClient (gRPC either way)."""
        if self.QDRANT_URL:
            return {"url": self.QDRANT_URL, "api_key": self.QDRANT_API_KEY or None, "prefer_grpc": True}
        return {"host": self.QDRANT_HOST, "grpc_port": self.QDRANT_GRPC_PORT, "prefer_grpc": True}

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()
