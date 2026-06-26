from pydantic import BaseModel, Field


class SearchQueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=5, le=20)


class ProductSearchResult(BaseModel):
    id: str | int
    title: str
    description: str
    price: float
    category: str
    score: float


class SearchResponseEnvelope(BaseModel):
    results: list[ProductSearchResult]
    latency_ms: float


# ── Diagnostic fields for /ask endpoint ──────────────────

class RetrievedProduct(BaseModel):
    id: str | int
    title: str
    category: str
    price: float
    rrf_score: float


class TimingBreakdown(BaseModel):
    cache_lookup_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


class AskResponseEnvelope(BaseModel):
    answer: str
    source: str
    retrieved_products: list[RetrievedProduct]
    timing: TimingBreakdown