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