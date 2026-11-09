"""In-process stand-ins for Qdrant search, Redis and Groq, shared by the API tests."""
from redis.exceptions import ConnectionError as RedisConnectionError

from src.backend.services.llm import UNAVAILABLE_MESSAGE, LLMResult


class FakeSearch:
    def __init__(self, n_results: int = 20):
        self.limits: list[int] = []
        self.calls: list[dict] = []
        self.n_results = n_results

    async def search(self, query, limit=5, mode=None, filters=None, prefer_size=None):
        self.limits.append(limit)
        self.calls.append({"query": query, "filters": filters, "prefer_size": prefer_size})
        return [
            {"id": str(100 + i), "score": 1 / (60 + i + 1), "title": f"P{i}",
             "description": "d", "price": 9.99, "category": "c"}
            for i in range(min(limit, self.n_results))
        ]

    async def facet_counts(self, key="category", limit=100):
        return [{"value": "Furniture", "count": 3}, {"value": "Rugs", "count": 1}][:limit]

    async def ready(self):
        return 42

    async def active_collection(self):
        return "test_products_v1"


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.store, self.fail = {}, fail

    async def get(self, k):
        if self.fail:
            raise RedisConnectionError("down")
        return self.store.get(k)

    async def set(self, k, v, ex=None, nx=False):
        if self.fail:
            raise RedisConnectionError("down")
        if nx and k in self.store:
            return None
        self.store[k] = v
        return True

    async def lpush(self, k, v):
        if self.fail:
            raise RedisConnectionError("down")
        self.store.setdefault(k, []).insert(0, v)
        return len(self.store[k])

    async def ltrim(self, k, start, end):
        self.store[k] = self.store.get(k, [])[start:end + 1]
        return True

    async def ping(self):
        if self.fail:
            raise RedisConnectionError("down")
        return True

    async def aclose(self):
        pass


class FakeLLM:
    def __init__(self, ok: bool = True):
        self.ok, self.calls, self.contexts = ok, 0, []

    async def generate_answer(self, query, context):
        self.calls += 1
        self.contexts.append(context)
        return LLMResult("answer", True, 900, 150) if self.ok else LLMResult(UNAVAILABLE_MESSAGE, False)
