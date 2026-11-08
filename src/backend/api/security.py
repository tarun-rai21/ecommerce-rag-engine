"""API-key authentication and per-client rate limiting for /api/v1."""
import hashlib
import hmac
import math
import time
from collections.abc import Callable

from fastapi import Header, Request

from config.settings import settings
from src.backend.api.errors import ApiError


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Checks X-API-Key against API_KEYS. With API_KEYS empty, auth is off (local development)."""
    keys = settings.api_keys
    if not keys:
        return
    # compare_digest: the time taken doesn't reveal how much of a key matched.
    if not x_api_key or not any(hmac.compare_digest(x_api_key.encode(), k.encode()) for k in keys):
        raise ApiError(401, "unauthorized", "Missing or invalid API key. Send it in the X-API-Key header.")


def client_id(request: Request) -> str:
    """Rate-limit identity: the API key when one is sent (hashed, never stored raw), else the IP.
    Behind a proxy, run uvicorn with --proxy-headers so the IP is the caller's."""
    key = request.headers.get("X-API-Key")
    if key:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


class RateLimiter:
    """Token bucket per (scope, client): up to `per_minute` requests in a burst, refilled
    at per_minute / 60 per second. In memory, so limits are per process; several
    instances would need a shared store such as Redis."""

    MAX_BUCKETS = 10_000

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}

    def check(self, scope: str, client: str, per_minute: int) -> float | None:
        """None if allowed (and a token is spent), else seconds until the next token."""
        if per_minute <= 0:
            return None
        now = self.clock()
        tokens, last = self._buckets.get((scope, client), (float(per_minute), now))
        tokens = min(float(per_minute), tokens + (now - last) * per_minute / 60)
        if tokens >= 1:
            self._buckets[(scope, client)] = (tokens - 1, now)
            if len(self._buckets) > self.MAX_BUCKETS:
                self._evict(now)
            return None
        self._buckets[(scope, client)] = (tokens, now)
        return (1 - tokens) * 60 / per_minute

    def _evict(self, now: float) -> None:
        # A bucket idle for a minute or more has refilled completely: forgetting it changes nothing.
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < 60}

    def reset(self) -> None:
        self._buckets.clear()


rate_limiter = RateLimiter()


def rate_limit(scope: str, setting: str):
    """Dependency enforcing the limit named by `setting`, read per request so it can change at runtime."""
    async def _check(request: Request) -> None:
        per_minute = getattr(settings, setting)
        wait = rate_limiter.check(scope, client_id(request), per_minute)
        if wait is not None:
            retry = max(1, math.ceil(wait))
            raise ApiError(429, "rate_limited", f"Too many requests: the limit is {per_minute} per minute. "
                           f"Try again in {retry} s.", headers={"Retry-After": str(retry)})
    return _check
