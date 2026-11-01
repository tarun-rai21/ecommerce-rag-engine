import time
from fastapi import APIRouter, Depends, Request
from loguru import logger
from pydantic import ValidationError

from src.backend.models.api_schemas import (
    SearchQueryRequest,
    ProductSearchResult,
    SearchResponseEnvelope,
    RetrievedProduct,
    TimingBreakdown,
    AskResponseEnvelope,
)
from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService, REFUSAL_MESSAGE

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


# Descriptions can run to 2,000 chars; five full ones would cost ~2.5k prompt tokens
# per request against an 8k tokens/min free-tier limit.
CONTEXT_DESCRIPTION_CHARS = 400


def build_context(results: list[dict]) -> str:
    lines = []
    for r in results:
        fields = [
            f"Title: {r.get('title', '')}",
            f"Category: {r.get('product_class') or r.get('category', '')}",
            f"Price: ${float(r.get('price', 0.0)):.2f}",
        ]
        if r.get("rating") is not None:
            fields.append(f"Rating: {r['rating']:.1f}/5 ({r.get('review_count') or 0} reviews)")
        if r.get("features"):
            fields.append("Attributes: " + ", ".join(f"{k}: {v}" for k, v in r["features"].items()))
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
    raw_results = await service.search(query=request.q, limit=request.limit)

    results = [
        ProductSearchResult(
            id=r["id"],
            title=r.get("title", ""),
            description=r.get("description", ""),
            price=r.get("price", 0.0),
            category=r.get("category", ""),
            score=r["score"],
        )
        for r in raw_results
    ]
    return SearchResponseEnvelope(results=results, latency_ms=_ms(start))


# ── /ask Endpoint ─────────────────────────────────────────
@router.post("/ask", response_model=AskResponseEnvelope)
async def ask_products(
    request: SearchQueryRequest,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> AskResponseEnvelope:
    total_start = time.perf_counter()

    # Step 1: Cache check. The full response (answer + products) is cached, so a
    # hit is indistinguishable from a miss apart from `source` and `timing`.
    cache_start = time.perf_counter()
    cache_key = cache_service.make_key("ask", request.q, limit=request.limit)
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

    # Step 2: Hybrid search
    logger.info(f"[CACHE MISS] query='{request.q}' — running hybrid search.")
    search_start = time.perf_counter()
    raw_results = await search_service.search(query=request.q, limit=request.limit)
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
        )

    # Step 3: LLM generation over the retrieved context
    llm_start = time.perf_counter()
    llm_result = await llm_service.generate_answer(query=request.q, context=build_context(raw_results))
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
                rrf_score=round(r["score"], 6),
            )
            for r in raw_results
        ],
        timing=TimingBreakdown(
            cache_lookup_ms=cache_ms, search_ms=search_ms, llm_ms=llm_ms,
            total_ms=_ms(total_start),
        ),
    )

    # Step 4: Cache write — successful generations only, never fallback messages.
    if llm_result.ok and await cache_service.set(cache_key, envelope.model_dump_json()):
        logger.info(f"[CACHE SET] key={cache_key}")

    return envelope
