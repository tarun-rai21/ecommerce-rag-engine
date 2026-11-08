import time
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from loguru import logger
from pydantic import ValidationError

from src.backend.models.api_schemas import (
    SearchQueryRequest,
    ProductSearchResult,
    SearchResponseEnvelope,
    RetrievedProduct,
    TimingBreakdown,
    AskResponseEnvelope,
    FacetResponse,
    FacetValue,
)
from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService, REFUSAL_MESSAGE
from src.backend.services.query_parser import ParsedQuery, SearchFilters, parse_query
from src.ingestion.schemas import size_token

router = APIRouter()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


# ── Dependency Injection ──────────────────────────────────
def get_search_service(request: Request) -> ProductSearchService:
    return request.app.state.search_service


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


# ── Query interpretation ──────────────────────────────────
def resolve_query(request: SearchQueryRequest) -> ParsedQuery:
    """Text for retrieval + the filters to apply: constraints parsed from the
    query text, overridden field by field by explicit request filters."""
    parsed = parse_query(request.q) if request.parse_filters else ParsedQuery(
        text=request.q.strip(), filters=SearchFilters())
    explicit = request.filters or SearchFilters()
    return ParsedQuery(text=parsed.text, filters=explicit.merged_over(parsed.filters), size=parsed.size)


def describe_filters(filters: SearchFilters) -> str:
    parts = []
    if filters.min_price is not None and filters.max_price is not None:
        parts.append(f"price between ${filters.min_price:.2f} and ${filters.max_price:.2f}")
    elif filters.max_price is not None:
        parts.append(f"price at most ${filters.max_price:.2f}")
    elif filters.min_price is not None:
        parts.append(f"price at least ${filters.min_price:.2f}")
    if filters.categories:
        parts.append("category in " + ", ".join(filters.categories))
    if filters.min_rating is not None:
        parts.append(f"rating at least {filters.min_rating:g}/5")
    return "; ".join(parts)


# Descriptions can run to 2,000 chars; five full ones would cost ~2.5k prompt tokens
# per request against an 8k tokens/min free-tier limit.
CONTEXT_DESCRIPTION_CHARS = 400
CONTEXT_MAX_SIZES = 20


def build_context(results: list[dict], filters: SearchFilters | None = None, prefer_size: str | None = None) -> str:
    lines = []
    if filters and not filters.is_empty():
        # Products were retrieved under these hard filters; telling the model avoids
        # refusals like "I can't confirm these are under $300".
        lines.append(f"(All products below already satisfy the shopper's filters: {describe_filters(filters)}.)")
    sold_in_size = [
        f"{r.get('title', '')} ({', '.join(s for s in r['sizes'] if size_token(s) == prefer_size)})"
        for r in results if any(size_token(s) == prefer_size for s in r.get("sizes") or [])
    ]
    if sold_in_size:
        # A rule in the system prompt didn't stop the model dropping near sizes such as
        # 7' 10" x 10' 2"; naming the products here is more direct.
        lines.append(f"(The shopper asked for {prefer_size}. Sold at or close to that size: {'; '.join(sold_in_size)}. "
                     "Show all of these, each with the size given here. The catalog may hold others, "
                     "so never call these the only ones.)")
    for r in results:
        fields = [
            f"Title: {r.get('title', '')}",
            f"Category: {r.get('product_class') or r.get('category', '')}",
            f"Price: ${float(r.get('price', 0.0)):.2f}",
        ]
        if r.get("rating") is not None:
            fields.append(f"Rating: {r['rating']:.1f}/5 ({r.get('review_count') or 0} reviews)")
        if r.get("features"):
            # Sorted: payload maps arrive in a different order per process, which would
            # change the prompt (and defeat response caching) for the same product.
            fields.append("Attributes: " + ", ".join(f"{k}: {v}" for k, v in sorted(r["features"].items())))
        if r.get("sizes"):
            shown = r["sizes"][:CONTEXT_MAX_SIZES]
            more = len(r["sizes"]) - len(shown)
            fields.append("Available sizes: " + ", ".join(shown) + (f" and {more} more" if more else ""))
            # Catalog sizes rarely read as the shopper's: a 7' 10" x 10' 2" rug is sold as an 8x10.
            matching = [s for s in r["sizes"] if size_token(s) == prefer_size]
            if matching:
                fields.append(f"Matches the requested {prefer_size} size: " + ", ".join(matching))
        if r.get("dimensions"):
            fields.append("Overall dimensions (inches): " + ", ".join(f"{k} {v}" for k, v in sorted(r["dimensions"].items())))
        description = (r.get("description") or "")[:CONTEXT_DESCRIPTION_CHARS]
        fields.append(f"Description: {description}")
        lines.append("- " + " | ".join(fields))
    return "\n".join(lines)


# ── /search Endpoint ──────────────────────────────────────
@router.post("/search", response_model=SearchResponseEnvelope)
async def search_products(
    request: SearchQueryRequest,
    service: ProductSearchService = Depends(get_search_service),
) -> SearchResponseEnvelope:
    start = time.perf_counter()
    resolved = resolve_query(request)
    raw_results = await service.search(
        query=resolved.text, limit=request.limit, filters=resolved.filters, prefer_size=resolved.size,
    )

    results = [
        ProductSearchResult(
            id=r["id"],
            title=r.get("title", ""),
            description=r.get("description", ""),
            price=r.get("price", 0.0),
            category=r.get("category", ""),
            rating=r.get("rating"),
            score=r["score"],
        )
        for r in raw_results
    ]
    return SearchResponseEnvelope(
        results=results, latency_ms=_ms(start),
        interpreted_query=resolved.text, applied_filters=resolved.filters, preferred_size=resolved.size,
    )


# ── /facets Endpoint ──────────────────────────────────────
@router.get("/facets", response_model=FacetResponse)
async def facets(
    key: Literal["category", "product_class"] = Query(default="category"),
    limit: int = Query(default=100, ge=1, le=1000),
    service: ProductSearchService = Depends(get_search_service),
) -> FacetResponse:
    values = await service.facet_counts(key=key, limit=limit)
    return FacetResponse(key=key, values=[FacetValue(**v) for v in values])


# ── /ask Endpoint ─────────────────────────────────────────
@router.post("/ask", response_model=AskResponseEnvelope)
async def ask_products(
    request: SearchQueryRequest,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> AskResponseEnvelope:
    total_start = time.perf_counter()
    resolved = resolve_query(request)

    # Step 1: Cache check. The full response (answer + products) is cached, so a
    # hit is indistinguishable from a miss apart from `source` and `timing`.
    # Filters are part of the key: the same words with a different price cap
    # must not share an answer.
    cache_start = time.perf_counter()
    cache_key = cache_service.make_key(
        "ask", request.q, limit=request.limit, filters=resolved.filters.model_dump_json(),
    )
    cached = await cache_service.get(cache_key)
    cache_ms = _ms(cache_start)

    if cached:
        try:
            envelope = AskResponseEnvelope.model_validate_json(cached)
            logger.info(f"[CACHE HIT] query='{request.q}'")
            return envelope.model_copy(update={
                "source": "cache",
                "timing": TimingBreakdown(
                    cache_lookup_ms=cache_ms, search_ms=0.0, llm_ms=0.0,
                    total_ms=_ms(total_start),
                ),
            })
        except ValidationError:
            logger.warning(f"[CACHE CORRUPT] key={cache_key} — ignoring entry.")

    # Step 2: Hybrid search under hard filters
    logger.info(f"[CACHE MISS] query='{request.q}' → text='{resolved.text}' filters={resolved.filters.model_dump(exclude_defaults=True)}")
    search_start = time.perf_counter()
    raw_results = await search_service.search(
        query=resolved.text, limit=request.limit, filters=resolved.filters, prefer_size=resolved.size,
    )
    search_ms = _ms(search_start)

    if not raw_results:
        return AskResponseEnvelope(
            answer=REFUSAL_MESSAGE,
            source="no_results",
            retrieved_products=[],
            timing=TimingBreakdown(
                cache_lookup_ms=cache_ms, search_ms=search_ms, llm_ms=0.0,
                total_ms=_ms(total_start),
            ),
            interpreted_query=resolved.text,
            applied_filters=resolved.filters,
            preferred_size=resolved.size,
        )

    # Step 3: LLM generation over the retrieved context
    llm_start = time.perf_counter()
    llm_result = await llm_service.generate_answer(
        query=request.q, context=build_context(raw_results, resolved.filters, resolved.size),
    )
    llm_ms = _ms(llm_start)

    envelope = AskResponseEnvelope(
        answer=llm_result.answer,
        source="llm" if llm_result.ok else "llm_error",
        retrieved_products=[
            RetrievedProduct(
                id=r["id"],
                title=r.get("title", ""),
                category=r.get("category", ""),
                price=r.get("price", 0.0),
                rating=r.get("rating"),
                rrf_score=round(r["score"], 6),
            )
            for r in raw_results
        ],
        timing=TimingBreakdown(
            cache_lookup_ms=cache_ms, search_ms=search_ms, llm_ms=llm_ms,
            total_ms=_ms(total_start),
        ),
        interpreted_query=resolved.text,
        applied_filters=resolved.filters,
        preferred_size=resolved.size,
    )

    # Step 4: Cache write — successful generations only, never fallback messages.
    if llm_result.ok and await cache_service.set(cache_key, envelope.model_dump_json()):
        logger.info(f"[CACHE SET] key={cache_key}")

    return envelope
