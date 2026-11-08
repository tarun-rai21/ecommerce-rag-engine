import re
import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# A caller-supplied id is reused only if it's short and plain, so it can't inject into logs.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Gives every request an id (the caller's X-Request-ID if valid, else a new one),
    logs method, path, status and latency with it, and returns it as X-Request-ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[{request.method}] {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Latency: {duration_ms:.2f}ms | request_id={request_id}"
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response
