import json
import time
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import ValidationError

from config.settings import settings
from src.backend.api.errors import ApiError
from src.backend.api.security import rate_limit
from src.backend.metrics import ANSWERS, FEEDBACK, LLM_TOKENS, observe_stage
from src.backend.models.api_schemas import (
    AskResponseEnvelope,
    FacetResponse,
    FacetValue,
    FeedbackRequest,
    FeedbackResponse,
    ProductSearchResult,
    RetrievedProduct,
    SearchQueryRequest,
    SearchResponseEnvelope,
    TimingBreakdown,
    TokenUsage,
)
from src.backend.services.cache import CacheService
from src.backend.services.feedback import AlreadyRated, FeedbackStore, FeedbackUnavailable, UnknownRequest
from src.backend.services.llm import (
    REFUSAL_MESSAGE,
    UNAVAILABLE_MESSAGE,
    LLMService,
    LLMUnavailable,
    StreamUsage,
    answer_fingerprint,
)
from src.backend.services.query_parser import ParsedQuery, SearchFilters, parse_query
from src.backend.services.search import ProductSearchService
from src.backend.services.stream_cleaner import StreamCleaner
from src.ingestion.schemas import size_token

router = APIRouter()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


SEARCH_UNAVAILABLE = "Product search is temporarily unavailable. Please try again shortly."


async def _search_or_503(call):
    """Run a Qdrant-backed call; a failure there becomes a 503 rather than a 500,
    since nothing can be answered without the catalog."""
    try:
        return await call
    except ValueError:
        raise  # a programming error (e.g. bad mode), not an outage
    except Exception as e:
        logger.opt(exception=e).error("Search backend failed")
        raise ApiError(503, "search_unavailable", SEARCH_UNAVAILABLE) from e


# ── Dependency Injection ──────────────────────────────────
def get_search_service(request: Request) -> ProductSearchService:
    return request.app.state.search_service


def get_cache_service(request: Request) -> CacheService:
    return request.app.state.cache_service


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_feedback_store(request: Request) -> FeedbackStore:
    return request.app.state.feedback_store


def _interaction(query: str, interpreted: str, filters: SearchFilters, size: str | None,
                 product_ids: list, answer: str, source: str, collection: str) -> dict:
    """What the shopper saw for one request, kept so a later rating can be analysed against it."""
    return {
        "query": query, "interpreted_query": interpreted,
        "filters": filters.model_dump(exclude_defaults=True), "preferred_size": size,
        "retrieved_ids": [str(i) for i in product_ids], "answer": answer, "source": source,
        "model": settings.LLM_MODEL, "collection": collection,
        "cache_version": settings.CACHE_VERSION,
        "answered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def answer_cache_key(cache_service: CacheService, request: SearchQueryRequest, resolved: ParsedQuery,
                     index: str) -> str:
    """Everything that changes the answer: the query, how it was interpreted (filters,
    size, whether parsing was on), the result count, the index version, and the model
    and prompt. /ask and /ask/stream share it."""
    return cache_service.make_key(
        "ask", request.q, limit=request.limit, filters=resolved.filters.model_dump_json(),
        parse=request.parse_filters, size=resolved.size, index=index, answer=answer_fingerprint(),
    )


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
@router.post("/search", response_model=SearchResponseEnvelope,
             dependencies=[Depends(rate_limit("search", "RATE_LIMIT_PER_MINUTE"))])
async def search_products(
    request: SearchQueryRequest,
    service: ProductSearchService = Depends(get_search_service),
) -> SearchResponseEnvelope:
    start = time.perf_counter()
    resolved = resolve_query(request)
    raw_results = await _search_or_503(service.search(
        query=resolved.text, limit=request.limit, filters=resolved.filters, prefer_size=resolved.size,
    ))
    observe_stage("search", _ms(start))

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
@router.get("/facets", response_model=FacetResponse,
            dependencies=[Depends(rate_limit("search", "RATE_LIMIT_PER_MINUTE"))])
async def facets(
    key: Literal["category", "product_class"] = Query(default="category"),
    limit: int = Query(default=100, ge=1, le=1000),
    service: ProductSearchService = Depends(get_search_service),
) -> FacetResponse:
    values = await _search_or_503(service.facet_counts(key=key, limit=limit))
    return FacetResponse(key=key, values=[FacetValue(**v) for v in values])


# ── /ask Endpoint ─────────────────────────────────────────
@router.post("/ask", response_model=AskResponseEnvelope,
             dependencies=[Depends(rate_limit("ask", "ASK_RATE_LIMIT_PER_MINUTE"))])
async def ask_products(
    request: SearchQueryRequest,
    http_request: Request,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
    feedback: FeedbackStore = Depends(get_feedback_store),
) -> AskResponseEnvelope:
    envelope = await _answer(request, search_service, cache_service, llm_service)
    observe_stage("total", envelope.timing.total_ms)
    await feedback.record_interaction(getattr(http_request.state, "request_id", None), _interaction(
        request.q, envelope.interpreted_query, envelope.applied_filters, envelope.preferred_size,
        [p.id for p in envelope.retrieved_products], envelope.answer, envelope.source,
        await search_service.active_collection(),
    ))
    return envelope


async def _answer(
    request: SearchQueryRequest,
    search_service: ProductSearchService,
    cache_service: CacheService,
    llm_service: LLMService,
) -> AskResponseEnvelope:
    total_start = time.perf_counter()
    resolved = resolve_query(request)

    # Step 1: Cache check. The full response (answer + products) is cached, so a
    # hit is indistinguishable from a miss apart from `source` and `timing`.
    # Filters are part of the key: the same words with a different price cap
    # must not share an answer.
    cache_start = time.perf_counter()
    # The active index version is part of the key: after an alias switch, answers built
    # on the old index are no longer served.
    cache_key = answer_cache_key(cache_service, request, resolved, await search_service.active_collection())
    cached = await cache_service.get(cache_key)
    cache_ms = _ms(cache_start)
    observe_stage("cache_lookup", cache_ms)

    if cached:
        try:
            envelope = AskResponseEnvelope.model_validate_json(cached)
            logger.info(f"[CACHE HIT] query='{request.q}'")
            ANSWERS.labels("cache").inc()
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
    raw_results = await _search_or_503(search_service.search(
        query=resolved.text, limit=request.limit, filters=resolved.filters, prefer_size=resolved.size,
    ))
    search_ms = _ms(search_start)
    observe_stage("search", search_ms)

    if not raw_results:
        ANSWERS.labels("no_results").inc()
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
    observe_stage("llm", llm_ms)
    ANSWERS.labels("llm" if llm_result.ok else "llm_error").inc()
    if llm_result.prompt_tokens is not None:
        LLM_TOKENS.labels("prompt").inc(llm_result.prompt_tokens)
    if llm_result.completion_tokens is not None:
        LLM_TOKENS.labels("completion").inc(llm_result.completion_tokens)

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
        usage=TokenUsage(prompt_tokens=llm_result.prompt_tokens, completion_tokens=llm_result.completion_tokens)
        if llm_result.prompt_tokens is not None and llm_result.completion_tokens is not None else None,
    )

    # Step 4: Cache write — complete, successful generations only: never fallback
    # messages, and never an answer cut off at max_tokens.
    if llm_result.ok and not llm_result.truncated and await cache_service.set(cache_key, envelope.model_dump_json()):
        logger.info(f"[CACHE SET] key={cache_key}")

    return envelope


# ── /ask/stream Endpoint (Server-Sent Events) ─────────────
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _products(raw_results: list[dict]) -> list[RetrievedProduct]:
    return [
        RetrievedProduct(id=r["id"], title=r.get("title", ""), category=r.get("category", ""),
                         price=r.get("price", 0.0), rating=r.get("rating"), rrf_score=round(r["score"], 6))
        for r in raw_results
    ]


def _event_stream(events: AsyncIterator[str]) -> StreamingResponse:
    # no-cache + X-Accel-Buffering: stop proxies from holding the stream back.
    return StreamingResponse(events, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/ask/stream", dependencies=[Depends(rate_limit("ask", "ASK_RATE_LIMIT_PER_MINUTE"))],
             response_class=StreamingResponse,
             responses={200: {"content": {"text/event-stream": {}},
                              "description": "Events: meta, token (repeated), done; error on a mid-stream failure."}})
async def ask_products_stream(
    request: SearchQueryRequest,
    http_request: Request,
    search_service: ProductSearchService = Depends(get_search_service),
    cache_service: CacheService = Depends(get_cache_service),
    llm_service: LLMService = Depends(get_llm_service),
    feedback: FeedbackStore = Depends(get_feedback_store),
) -> StreamingResponse:
    """/ask as a stream. Same cache, filters and answer rules; the answer arrives as it is
    written, released one complete sentence or table row at a time (see StreamCleaner).

    Event order: `meta` (interpretation, products, request_id, source), `token` x N
    ({"text"}), then `done` (source, timing, usage). A failure before streaming starts is
    a normal JSON error; one after it starts is an `error` event followed by `done`."""
    total_start = time.perf_counter()
    resolved = resolve_query(request)
    request_id = getattr(http_request.state, "request_id", None)
    meta = {"request_id": request_id, "interpreted_query": resolved.text,
            "applied_filters": resolved.filters.model_dump(), "preferred_size": resolved.size}

    cache_start = time.perf_counter()
    cache_key = answer_cache_key(cache_service, request, resolved, await search_service.active_collection())
    cached = await cache_service.get(cache_key)
    cache_ms = _ms(cache_start)
    observe_stage("cache_lookup", cache_ms)
    envelope = None
    if cached:
        try:
            envelope = AskResponseEnvelope.model_validate_json(cached)
        except ValidationError:
            logger.warning(f"[CACHE CORRUPT] key={cache_key} — ignoring entry.")

    if envelope is not None:
        ANSWERS.labels("cache").inc()

        async def from_cache() -> AsyncIterator[str]:
            yield _sse("meta", {**meta, "source": "cache",
                                "retrieved_products": [p.model_dump() for p in envelope.retrieved_products]})
            yield _sse("token", {"text": envelope.answer})
            timing = TimingBreakdown(cache_lookup_ms=cache_ms, search_ms=0.0, llm_ms=0.0,
                                     total_ms=_ms(total_start))
            yield _sse("done", {"source": "cache", "timing": timing.model_dump(), "usage": None})
            observe_stage("total", timing.total_ms)
            await feedback.record_interaction(request_id, _interaction(
                request.q, resolved.text, resolved.filters, resolved.size,
                [p.id for p in envelope.retrieved_products], envelope.answer, "cache",
                await search_service.active_collection()))
        return _event_stream(from_cache())

    search_start = time.perf_counter()
    raw_results = await _search_or_503(search_service.search(
        query=resolved.text, limit=request.limit, filters=resolved.filters, prefer_size=resolved.size,
    ))
    search_ms = _ms(search_start)
    observe_stage("search", search_ms)
    products = _products(raw_results)

    async def generate() -> AsyncIterator[str]:
        if not raw_results:
            ANSWERS.labels("no_results").inc()
            yield _sse("meta", {**meta, "source": "no_results", "retrieved_products": []})
            yield _sse("token", {"text": REFUSAL_MESSAGE})
            timing = TimingBreakdown(cache_lookup_ms=cache_ms, search_ms=search_ms, llm_ms=0.0,
                                     total_ms=_ms(total_start))
            yield _sse("done", {"source": "no_results", "timing": timing.model_dump(), "usage": None})
            observe_stage("total", timing.total_ms)
            await feedback.record_interaction(request_id, _interaction(
                request.q, resolved.text, resolved.filters, resolved.size, [], REFUSAL_MESSAGE, "no_results", await search_service.active_collection()))
            return

        yield _sse("meta", {**meta, "source": "llm", "retrieved_products": [p.model_dump() for p in products]})
        usage, cleaner, source = StreamUsage(), StreamCleaner(), "llm"
        llm_start, first_token = time.perf_counter(), None
        try:
            context = build_context(raw_results, resolved.filters, resolved.size)
            async for delta in llm_service.stream_answer(request.q, context, usage):
                text = cleaner.feed(delta)
                if text:
                    if first_token is None:
                        first_token = _ms(llm_start)
                        observe_stage("llm_first_token", first_token)
                    yield _sse("token", {"text": text})
            tail = cleaner.finish()
            if tail:
                yield _sse("token", {"text": tail})
            if not cleaner.text:  # nothing to show (e.g. all tokens spent on reasoning): as /ask does
                logger.error("LLM stream produced no answer text")
                source = "llm_error"
                yield _sse("token", {"text": UNAVAILABLE_MESSAGE})
        except LLMUnavailable as e:
            logger.error(f"LLM stream unavailable: {e}")
            source = "llm_error"
            yield _sse("token", {"text": UNAVAILABLE_MESSAGE})
        except Exception as e:  # failed mid-stream: the shopper already has part of an answer
            logger.opt(exception=e).error("LLM stream failed mid-answer")
            source = "llm_error"
            yield _sse("error", {"code": "llm_failed", "message": "The answer was interrupted. Please try again.",
                                 "request_id": request_id})
        llm_ms = _ms(llm_start)
        observe_stage("llm", llm_ms)
        ANSWERS.labels(source).inc()
        if usage.prompt_tokens is not None:
            LLM_TOKENS.labels("prompt").inc(usage.prompt_tokens)
        if usage.completion_tokens is not None:
            LLM_TOKENS.labels("completion").inc(usage.completion_tokens)

        timing = TimingBreakdown(cache_lookup_ms=cache_ms, search_ms=search_ms, llm_ms=llm_ms,
                                 total_ms=_ms(total_start))
        token_usage = (TokenUsage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)
                       if usage.prompt_tokens is not None and usage.completion_tokens is not None else None)
        yield _sse("done", {"source": source, "timing": timing.model_dump(),
                            "usage": token_usage.model_dump() if token_usage else None,
                            "time_to_first_token_ms": first_token})
        observe_stage("total", timing.total_ms)
        answer = cleaner.text if source == "llm" else (cleaner.text or UNAVAILABLE_MESSAGE)
        await feedback.record_interaction(request_id, _interaction(
            request.q, resolved.text, resolved.filters, resolved.size, [p.id for p in products], answer, source,
            await search_service.active_collection()))
        if usage.truncated:
            logger.warning(f"LLM answer cut off at max_tokens for query: '{request.q}'")
        # The same entry /ask writes, so either endpoint can serve it; never a cut-off answer.
        if source == "llm" and cleaner.text and not usage.truncated:
            stored = AskResponseEnvelope(
                answer=cleaner.text, source="llm", retrieved_products=products, timing=timing,
                interpreted_query=resolved.text, applied_filters=resolved.filters,
                preferred_size=resolved.size, usage=token_usage,
            )
            await cache_service.set(cache_key, stored.model_dump_json())

    return _event_stream(generate())


# ── /feedback Endpoint ────────────────────────────────────
@router.post("/feedback", response_model=FeedbackResponse, status_code=201,
             dependencies=[Depends(rate_limit("search", "RATE_LIMIT_PER_MINUTE"))])
async def submit_feedback(
    body: FeedbackRequest,
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    """Rate an answer (up/down, optional comment) by the request id it was served with
    (the X-Request-ID header, or `request_id` in the stream's meta event)."""
    days = settings.FEEDBACK_RETENTION_DAYS
    try:
        await store.add(body.request_id, body.rating, body.comment)
    except UnknownRequest as e:
        raise ApiError(404, "unknown_request",
                       f"No answer with this request id to rate. Answers can be rated for {days} days.") from e
    except AlreadyRated as e:
        raise ApiError(409, "already_rated", "This answer has already been rated.") from e
    except FeedbackUnavailable as e:
        raise ApiError(503, "feedback_unavailable", "Feedback can't be saved right now. Please try later.") from e
    FEEDBACK.labels(body.rating).inc()
    logger.info(f"feedback rating={body.rating} for request_id={body.request_id}")  # comments aren't logged
    return FeedbackResponse()
