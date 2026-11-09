import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.backend.services.query_parser import SearchFilters


class SearchQueryRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=200)

    @field_validator("q", mode="before")
    @classmethod
    def _strip_control_characters(cls, v):
        # Control and format characters (null bytes, zero-width spaces, bidi overrides) carry
        # no meaning in a product search but can hide text from people reading logs or
        # prompts. Each becomes a space; the length check then applies to what remains.
        if not isinstance(v, str):
            return v
        cleaned = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in v)
        return " ".join(cleaned.split())
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
    preferred_size: str | None = None  # products in this size rank first, e.g. "8x10"


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


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int  # includes the model's reasoning tokens, which are billed


AnswerSource = Literal["llm", "cache", "no_results", "llm_error"]


class AskResponseEnvelope(BaseModel):
    answer: str
    source: AnswerSource
    retrieved_products: list[RetrievedProduct]
    timing: TimingBreakdown
    interpreted_query: str
    applied_filters: SearchFilters
    preferred_size: str | None = None
    # Tokens of the LLM call that produced this answer. A cache hit repeats them but
    # costs nothing; count cost only where source == "llm".
    usage: TokenUsage | None = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., pattern=r"^[A-Za-z0-9._-]{1,64}$")  # X-Request-ID of the answer
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    status: Literal["recorded"] = "recorded"


class FacetValue(BaseModel):
    value: str
    count: int


class FacetResponse(BaseModel):
    key: str
    values: list[FacetValue]
