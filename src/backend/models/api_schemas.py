from typing import Literal
from pydantic import BaseModel, Field

from src.backend.services.query_parser import SearchFilters


class SearchQueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)
    # Explicit facet filters (e.g. from UI controls). They override constraints
    # parsed from the query text field by field.
    filters: SearchFilters | None = None
    # Parse "under $300", "4+ stars" etc. out of `q` into filters.
    parse_filters: bool = True


class ProductSearchResult(BaseModel):
    id: str | int
    title: str
    description: str
    price: float
    category: str
    rating: float | None = None
    score: float


class SearchResponseEnvelope(BaseModel):
    results: list[ProductSearchResult]
    latency_ms: float
    interpreted_query: str          # text actually used for retrieval
    applied_filters: SearchFilters  # parsed + explicit, as applied in Qdrant


# ── Diagnostic fields for /ask endpoint ──────────────────

class RetrievedProduct(BaseModel):
    id: str | int
    title: str
    category: str
    price: float
    rating: float | None = None
    rrf_score: float


class TimingBreakdown(BaseModel):
    cache_lookup_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


AnswerSource = Literal["llm", "cache", "no_results", "llm_error"]


class AskResponseEnvelope(BaseModel):
    answer: str
    source: AnswerSource
    retrieved_products: list[RetrievedProduct]
    timing: TimingBreakdown
    interpreted_query: str
    applied_filters: SearchFilters


class FacetValue(BaseModel):
    value: str
    count: int


class FacetResponse(BaseModel):
    key: str
    values: list[FacetValue]
