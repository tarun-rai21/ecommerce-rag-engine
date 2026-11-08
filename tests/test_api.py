"""/search and /ask behaviour with in-process fakes (no Qdrant, Redis or Groq)."""
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from src.backend.main import app
from src.backend.services.cache import CacheService
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


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.store, self.fail = {}, fail

    async def get(self, k):
        if self.fail:
            raise RedisConnectionError("down")
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        if self.fail:
            raise RedisConnectionError("down")
        self.store[k] = v

    async def aclose(self):
        pass


class FakeLLM:
    def __init__(self, ok: bool = True):
        self.ok, self.calls, self.contexts = ok, 0, []

    async def generate_answer(self, query, context):
        self.calls += 1
        self.contexts.append(context)
        return LLMResult("answer", True, 900, 150) if self.ok else LLMResult(UNAVAILABLE_MESSAGE, False)


@pytest.fixture
def make_client():
    def _make(search=None, redis=None, llm=None):
        app.state.search_service = search or FakeSearch()
        app.state.cache_service = CacheService(client=redis or FakeRedis())
        app.state.llm_service = llm or FakeLLM()
        # Not used as a context manager, so the lifespan (real services) doesn't run.
        return TestClient(app, raise_server_exceptions=False)
    return _make


def test_cache_hit_returns_same_products(make_client):
    llm = FakeLLM()
    c = make_client(llm=llm)
    first = c.post("/api/v1/ask", json={"q": "bag", "limit": 3}).json()
    second = c.post("/api/v1/ask", json={"q": "  BAG ", "limit": 3}).json()
    assert first["source"] == "llm" and second["source"] == "cache"
    assert second["retrieved_products"] == first["retrieved_products"]
    assert len(second["retrieved_products"]) == 3
    assert llm.calls == 1
    assert first["usage"] == second["usage"] == {"prompt_tokens": 900, "completion_tokens": 150}


def test_cache_key_includes_limit(make_client):
    llm = FakeLLM()
    c = make_client(llm=llm)
    c.post("/api/v1/ask", json={"q": "bag", "limit": 3})
    r = c.post("/api/v1/ask", json={"q": "bag", "limit": 5}).json()
    assert r["source"] == "llm" and len(r["retrieved_products"]) == 5


def test_redis_down_fails_open(make_client):
    c = make_client(redis=FakeRedis(fail=True))
    r = c.post("/api/v1/ask", json={"q": "bag"})
    assert r.status_code == 200 and r.json()["source"] == "llm"


def test_llm_failure_is_flagged_and_not_cached(make_client):
    redis = FakeRedis()
    c = make_client(redis=redis, llm=FakeLLM(ok=False))
    r = c.post("/api/v1/ask", json={"q": "bag"}).json()
    assert r["source"] == "llm_error" and r["answer"] == UNAVAILABLE_MESSAGE
    assert r["retrieved_products"]  # products still returned
    assert redis.store == {}


def test_corrupt_cache_entry_is_ignored(make_client):
    redis = FakeRedis()
    redis.store[CacheService.make_key("ask", "bag", limit=5)] = "{not json"
    c = make_client(redis=redis)
    r = c.post("/api/v1/ask", json={"q": "bag"})
    assert r.status_code == 200 and r.json()["source"] == "llm"


def test_no_results(make_client):
    c = make_client(search=FakeSearch(n_results=0))
    r = c.post("/api/v1/ask", json={"q": "zzz"}).json()
    assert r["source"] == "no_results" and r["retrieved_products"] == []


@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/ask"])
def test_limit_is_respected(make_client, path):
    search = FakeSearch()
    c = make_client(search=search)
    assert c.post(path, json={"q": "bag", "limit": 10}).status_code == 200
    assert search.limits == [10]


@pytest.mark.parametrize("body", [
    {"q": "bag", "limit": 0}, {"q": "bag", "limit": -3}, {"q": "bag", "limit": 21}, {"q": ""},
])
def test_invalid_requests_rejected(make_client, body):
    c = make_client()
    assert c.post("/api/v1/search", json=body).status_code == 422


# ── Filters ───────────────────────────────────────────────
def test_constraints_parsed_from_text_become_filters(make_client):
    search, llm = FakeSearch(), FakeLLM()
    c = make_client(search=search, llm=llm)
    r = c.post("/api/v1/ask", json={"q": "grey area rug under $300", "limit": 3}).json()
    call = search.calls[-1]
    assert call["query"] == "grey area rug" and call["filters"].max_price == 300
    assert r["interpreted_query"] == "grey area rug" and r["applied_filters"]["max_price"] == 300
    assert "price at most $300.00" in llm.contexts[-1]


def test_explicit_filters_override_parsed(make_client):
    search = FakeSearch()
    c = make_client(search=search)
    body = {"q": "rug under $300", "filters": {"max_price": 150, "categories": ["Rugs"]}}
    r = c.post("/api/v1/search", json=body).json()
    f = search.calls[-1]["filters"]
    assert f.max_price == 150 and f.categories == ["Rugs"]
    assert r["applied_filters"]["max_price"] == 150


def test_parse_filters_can_be_disabled(make_client):
    search = FakeSearch()
    c = make_client(search=search)
    c.post("/api/v1/search", json={"q": "rug under $300", "parse_filters": False})
    assert search.calls[-1]["query"] == "rug under $300" and search.calls[-1]["filters"].is_empty()


def test_different_filters_do_not_share_cache(make_client):
    llm = FakeLLM()
    c = make_client(llm=llm)
    c.post("/api/v1/ask", json={"q": "rug", "filters": {"max_price": 100}})
    r = c.post("/api/v1/ask", json={"q": "rug", "filters": {"max_price": 200}}).json()
    assert r["source"] == "llm" and llm.calls == 2


def test_invalid_filter_range_rejected(make_client):
    c = make_client()
    r = c.post("/api/v1/search", json={"q": "rug", "filters": {"min_price": 500, "max_price": 100}})
    assert r.status_code == 422


def test_facets_endpoint(make_client):
    c = make_client()
    r = c.get("/api/v1/facets", params={"key": "category"}).json()
    assert r["key"] == "category" and r["values"][0] == {"value": "Furniture", "count": 3}
    assert c.get("/api/v1/facets", params={"key": "description"}).status_code == 422


def test_size_in_query_becomes_a_preference(make_client):
    search, llm = FakeSearch(), FakeLLM()
    c = make_client(search=search, llm=llm)
    r = c.post("/api/v1/ask", json={"q": "grey area rug 8 x 10", "limit": 3}).json()
    call = search.calls[-1]
    assert call["query"] == "grey area rug" and call["prefer_size"] == "8x10" and call["filters"].is_empty()
    assert r["preferred_size"] == "8x10" and r["interpreted_query"] == "grey area rug"
