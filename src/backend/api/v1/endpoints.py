import time
from fastapi import APIRouter, Depends
from loguru import logger

from src.backend.models.api_schemas import (
    SearchQueryRequest,
    ProductSearchResult,
    SearchResponseEnvelope,
)
from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService

router = APIRouter()

# ── Dependency Injection ──────────────────────────────────
async def get_search_service() -> ProductSearchService:
    service = ProductSearchService()
    try:
        yield service
    finally:
        await service.close()


def get_cache_service() -> CacheService:
    return CacheService()


def get_llm_service() -> LLMService:
    return LLMService()


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
@router.post("/ask")
async def ask_products(
    request: SearchQueryRequest,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> dict:

    # Step 1: Check cache
    cache_key = cache_service._generate_key("rag:cache", request.q)
    cached = await cache_service.get(cache_key)
    if cached:
        logger.info(f"[CACHE HIT] query='{request.q}'")
        return {"answer": cached, "source": "cache"}

    # Step 2: Hybrid search
    logger.info(f"[CACHE MISS] query='{request.q}' — running hybrid search.")
    raw_results = await search_service.search(query=request.q, limit=5)

    # Step 3: Construct context blob
    if not raw_results:
        return {
            "answer": "I cannot find a matching product matching those exact parameters in our catalog.",
            "source": "no_results",
        }

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
    answer = await llm_service.generate_answer(
        query=request.q,
        context=context,
    )

    # Step 5: Write to cache
    if not answer.startswith("I was unable"):
        await cache_service.set(cache_key, answer)
        logger.info(f"[CACHE SET] key={cache_key}")

    return {"answer": answer, "source": "llm"}