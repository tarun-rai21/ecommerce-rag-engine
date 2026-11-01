import asyncio
import hashlib
import time
from typing import Any, Awaitable, Callable

from loguru import logger
import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

from config.settings import settings


class CacheService:
    """Redis response cache that can never take the API down.

    - Fail-open: any error is logged and treated as a miss (get) or a no-op (set).
    - Bounded: every operation has a hard time budget (``CACHE_OP_TIMEOUT_SECONDS``).
    - Circuit breaker: after a failure, Redis is skipped for ``CACHE_COOLDOWN_SECONDS``
      so an outage costs one timeout, not one per request.
    """

    def __init__(
        self,
        client: aioredis.Redis | None = None,
        op_timeout: float | None = None,
        cooldown: float | None = None,
    ) -> None:
        self.client = client or aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            retry=Retry(NoBackoff(), 0),  # no client-side retries; the breaker handles outages
        )
        self.op_timeout = op_timeout if op_timeout is not None else settings.CACHE_OP_TIMEOUT_SECONDS
        self.cooldown = cooldown if cooldown is not None else settings.CACHE_COOLDOWN_SECONDS
        self._skip_until = 0.0
        logger.info(f"CacheService initialized at '{settings.REDIS_HOST}:{settings.REDIS_PORT}'")

    @staticmethod
    def make_key(namespace: str, query: str, **params: object) -> str:
        """Key = rag:<cache version>:<namespace>:<sorted params>:md5(normalised query).

        Normalisation (lowercase, trimmed, whitespace collapsed) makes
        'Waterproof  Bag' and 'waterproof bag' share an entry. Params such as
        ``limit`` are part of the key because they change the response."""
        normalized = " ".join(query.lower().split())
        digest = hashlib.md5(normalized.encode()).hexdigest()
        param_str = ",".join(f"{k}={params[k]}" for k in sorted(params))
        return f"rag:{settings.CACHE_VERSION}:{namespace}:{param_str}:{digest}"

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._skip_until

    async def _run(self, op: str, call: Callable[[], Awaitable[Any]]) -> tuple[bool, Any]:
        """Run a Redis call within the time budget. Returns (succeeded, result)."""
        if not self.available:
            return False, None
        try:
            return True, await asyncio.wait_for(call(), timeout=self.op_timeout)
        except (RedisError, OSError, TimeoutError) as e:
            self._skip_until = time.monotonic() + self.cooldown
            logger.warning(f"Cache {op} failed ({e!r}); bypassing cache for {self.cooldown:.0f}s.")
            return False, None

    async def get(self, key: str) -> str | None:
        _, value = await self._run("GET", lambda: self.client.get(key))
        return value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> bool:
        ok, _ = await self._run(
            "SET", lambda: self.client.set(key, value, ex=ttl_seconds or settings.CACHE_TTL_SECONDS)
        )
        return ok

    async def invalidate(self, key: str) -> None:
        if (await self._run("DELETE", lambda: self.client.delete(key)))[0]:
            logger.info(f"Cache invalidated for key: {key}")

    async def close(self) -> None:
        await self.client.aclose()
