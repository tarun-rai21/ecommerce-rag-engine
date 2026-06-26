import hashlib
import asyncio
from loguru import logger
import redis.asyncio as aioredis

from config.settings import settings


class CacheService:
    def __init__(self) -> None:
        self.client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )
        logger.info(f"CacheService initialized at '{settings.REDIS_HOST}:{settings.REDIS_PORT}'")

    def _generate_key(self, prefix: str, query: str) -> str:
        normalized = query.lower().strip()
        hash_digest = hashlib.md5(normalized.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"

    async def get(self, key: str) -> str | None:
        value = await self.client.get(key)
        return value

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        await self.client.set(key, value, ex=ttl_seconds)

    async def invalidate(self, key: str) -> None:
        await self.client.delete(key)
        logger.info(f"Cache invalidated for key: {key}")

    async def close(self) -> None:
        await self.client.aclose()