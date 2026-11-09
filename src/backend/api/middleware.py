import re
import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.backend.metrics import REQUEST_LATENCY, REQUESTS

# A caller-supplied id is reused only if it's short and plain, so it can't inject into logs.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _route_template(request: Request) -> str:
    """The full route template ("/api/v1/search"), never the raw path, so label values stay
    bounded: unknown paths (scanners, typos) all count as "unmatched".

    For a route in an included router, scope["route"] holds the inner template ("/search")
    without the prefix. The inner route matches the tail of the URL, so what comes before
    that tail is the prefix. Prefixes here are static ("/api/v1"), so this can't leak ids."""
    route = request.scope.get("route")
    if route is None:
        return "unmatched"
    path = request.scope["path"]
    for i, ch in enumerate(path):
        if ch == "/" and route.path_regex.match(path[i:]):
            return path[:i] + route.path
    return route.path


def _record(request: Request, status: int, seconds: float) -> None:
    route = _route_template(request)
    REQUESTS.labels(route, request.method, str(status)).inc()
    REQUEST_LATENCY.labels(route).observe(seconds)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Gives every request an id (the caller's X-Request-ID if valid, else a new one),
    logs method, path, status and latency with it, and returns it as X-Request-ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start_time = time.perf_counter()

        # Every log line written while handling this request carries its id.
        with logger.contextualize(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                _record(request, 500, time.perf_counter() - start_time)
                raise
            duration_ms = (time.perf_counter() - start_time) * 1000
            _record(request, response.status_code, duration_ms / 1000)
            logger.info(
                f"[{request.method}] {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Latency: {duration_ms:.2f}ms"
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response
