"""Prometheus metrics, served at /metrics.

Labels use route templates and fixed values only (never query text or ids), so the
number of time series stays small however much traffic arrives. The app runs one
uvicorn worker, so the default in-process registry is enough; several worker processes
would need prometheus_client's multiprocess mode.
"""
import hmac

from fastapi import APIRouter, Header
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, disable_created_metrics, generate_latest

from config.settings import settings
from src.backend.api.errors import ApiError

# No *_created series: they double the output and no dashboard uses them.
disable_created_metrics()

# Request-level, recorded by RequestContextMiddleware.
REQUESTS = Counter("rag_requests_total", "HTTP requests", ["route", "method", "status"])
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds", "End-to-end request latency in the server", ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

# Pipeline stages of /search and /ask.
STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds", "Latency of one pipeline stage", ["stage"],  # cache_lookup | search | llm
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
ANSWERS = Counter("rag_answers_total", "/ask responses by where the answer came from",
                  ["source"])  # llm | cache | no_results | llm_error
LLM_TOKENS = Counter("rag_llm_tokens_total", "Tokens billed by the LLM provider", ["type"])  # prompt | completion

# Cache.
CACHE_LOOKUPS = Counter("rag_cache_lookups_total", "Cache GETs by outcome",
                        ["result"])  # hit | miss | error | bypassed
CACHE_BREAKER_OPEN = Gauge("rag_cache_breaker_open", "1 while the circuit breaker is skipping Redis")

# Safeguards.
RATE_LIMITED = Counter("rag_rate_limited_total", "Requests refused by a rate limit", ["scope"])
AUTH_FAILURES = Counter("rag_auth_failures_total", "Requests refused for a missing or invalid API key")
FEEDBACK = Counter("rag_feedback_total", "Shopper ratings of answers", ["rating"])  # up | down


# Create every known label value at 0 now. A series that first appears with a non-zero
# value has no "before" sample, so rate() and increase() would miss its first events.
for _source in ("llm", "cache", "no_results", "llm_error"):
    ANSWERS.labels(_source)
for _type in ("prompt", "completion"):
    LLM_TOKENS.labels(_type)
for _result in ("hit", "miss", "error", "bypassed"):
    CACHE_LOOKUPS.labels(_result)
for _scope in ("search", "ask"):
    RATE_LIMITED.labels(_scope)
for _rating in ("up", "down"):
    FEEDBACK.labels(_rating)


def observe_stage(stage: str, milliseconds: float) -> None:
    STAGE_LATENCY.labels(stage).observe(milliseconds / 1000)


router = APIRouter(tags=["monitoring"])


@router.get("/metrics", include_in_schema=False)
async def metrics(authorization: str | None = Header(default=None)) -> Response:
    """Prometheus text format. With METRICS_TOKEN set, needs `Authorization: Bearer <token>`
    (Prometheus: `authorization: {credentials: <token>}` in the scrape config)."""
    token = settings.METRICS_TOKEN
    if token:
        sent = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(sent.encode(), token.encode()):
            raise ApiError(401, "unauthorized", "Missing or invalid metrics token.")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
