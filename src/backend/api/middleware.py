import time
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"[{request.method}] {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Latency: {duration_ms:.2f}ms"
        )

        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

        return response