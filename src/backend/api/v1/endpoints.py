import time
from fastapi import APIRouter, Depends, Request
from loguru import logger

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
from src.backend.services.llm import LLMService

router = APIRouter()


# ── Dependency Injection ──────────────────────────────────
def get_search_service(request: Request) -> ProductSearchService:
    return request.app.state.search_service


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


# ── /search Endpoint ──────────────────────────────────────
@router.post("/search", response_model=SearchResponseEnvelope)
async def search_products(
    request: SearchQueryRequest,
    service: ProductSearchService = Depends(get_search_service),
) -> SearchResponseEnvelope:
    start = time.perf_counter()

    raw_results = await service.search(
        query=request.q,
        limit=request.limit,
    )

    latency_ms = (time.perf_counter() - start) * 1000

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

    return SearchResponseEnvelope(
        results=results,
        latency_ms=round(latency_ms, 2),
    )


# ── /ask Endpoint ─────────────────────────────────────────
@router.post("/ask", response_model=AskResponseEnvelope)
async def ask_products(
    request: SearchQueryRequest,
    http_request: Request,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> AskResponseEnvelope:
    total_start = time.perf_counter()

    # Step 1: Cache check
    cache_start = time.perf_counter()
    cache_key = cache_service._generate_key("rag:cache", request.q)
    cached = await cache_service.get(cache_key)
    cache_ms = (time.perf_counter() - cache_start) * 1000

    if cached:
        logger.info(f"[CACHE HIT] query='{request.q}'")
        total_ms = (time.perf_counter() - total_start) * 1000
        return AskResponseEnvelope(
            answer=cached,
            source="cache",
            retrieved_products=[],
            timing=TimingBreakdown(
                cache_lookup_ms=round(cache_ms, 2),
                search_ms=0.0,
                llm_ms=0.0,
                total_ms=round(total_ms, 2),
            ),
        )

    # Step 2: Hybrid search
    logger.info(f"[CACHE MISS] query='{request.q}' — running hybrid search.")
    search_start = time.perf_counter()
    raw_results = await search_service.search(query=request.q, limit=5)
    search_ms = (time.perf_counter() - search_start) * 1000

    if not raw_results:
        total_ms = (time.perf_counter() - total_start) * 1000
        return AskResponseEnvelope(
            answer="I cannot find a matching product matching those exact parameters in our catalog.",
            source="no_results",
            retrieved_products=[],
            timing=TimingBreakdown(
                cache_lookup_ms=round(cache_ms, 2),
                search_ms=round(search_ms, 2),
                llm_ms=0.0,
                total_ms=round(total_ms, 2),
            ),
        )

    # Step 3: Build context blob
    context_lines = []
    for r in raw_results:
        context_lines.append(
            f"- Title: {r.get('title', '')} | "
            f"Category: {r.get('category', '')} | "
            f"Price: ${r.get('price', 0.0)} | "
            f"Description: {r.get('description', '')}"
        )
    context = "\n".join(context_lines)

    # Step 4: LLM generation
    llm_start = time.perf_counter()
    answer = await llm_service.generate_answer(query=request.q, context=context)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    # Step 5: Cache write
    if not answer.startswith("I was unable"):
        await cache_service.set(cache_key, answer)
        logger.info(f"[CACHE SET] key={cache_key}")

    total_ms = (time.perf_counter() - total_start) * 1000

    retrieved_products = [
        RetrievedProduct(
            id=r["id"],
            title=r.get("title", ""),
            category=r.get("category", ""),
            price=r.get("price", 0.0),
            rrf_score=round(r["score"], 6),
        )
        for r in raw_results
    ]

    return AskResponseEnvelope(
        answer=answer,
        source="llm",
        retrieved_products=retrieved_products,
        timing=TimingBreakdown(
            cache_lookup_ms=round(cache_ms, 2),
            search_ms=round(search_ms, 2),
            llm_ms=round(llm_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )