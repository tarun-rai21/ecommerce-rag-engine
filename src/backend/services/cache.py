import hashlib
import asyncio
from loguru import logger
import diskcache


CACHE_DIR = ".cache/rag_cache"


class CacheService:
    def __init__(self) -> None:
        self.cache = diskcache.Cache(CACHE_DIR)
        logger.info(f"CacheService initialized at '{CACHE_DIR}'")

    def _generate_key(self, prefix: str, query: str) -> str:
        normalized = query.lower().strip()
        hash_digest = hashlib.md5(normalized.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"

    async def get(self, key: str) -> str | None:
        value = await asyncio.to_thread(self.cache.get, key, None)
        return value

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        await asyncio.to_thread(self.cache.set, key, value, expire=ttl_seconds)

    async def invalidate(self, key: str) -> None:
        await asyncio.to_thread(self.cache.delete, key)
        logger.info(f"Cache invalidated for key: {key}")

    def close(self) -> None:
        self.cache.close()