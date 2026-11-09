"""Health probes, error envelope, API keys, rate limits and input guards."""
import pytest

from config.settings import settings
from src.backend.api.security import RateLimiter, rate_limiter
from tests.fakes import FakeRedis, FakeSearch


@pytest.fixture(autouse=True)
def fresh_limits(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "")
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(settings, "ASK_RATE_LIMIT_PER_MINUTE", 0)
    rate_limiter.reset()
    yield
    rate_limiter.reset()


class DownSearch(FakeSearch):
    async def search(self, *a, **kw):
        raise ConnectionError("qdrant unreachable")

    async def facet_counts(self, *a, **kw):
        raise ConnectionError("qdrant unreachable")

    async def ready(self):
        raise ConnectionError("qdrant unreachable")


# ── Health ────────────────────────────────────────────────
def test_healthz(make_client):
    assert make_client().get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_each_dependency(make_client):
    r = make_client(redis=FakeRedis(fail=True)).get("/readyz")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ready"  # Redis is optional
    assert body["checks"]["qdrant"] == {"ok": True, "points": 42, "collection": "test_products_v1"}
    assert body["checks"]["cache"] == {"ok": False, "error": "ConnectionError"}


def test_readyz_fails_without_qdrant(make_client):
    r = make_client(search=DownSearch()).get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["qdrant"] == {"ok": False, "error": "ConnectionError"}


def test_probes_need_no_api_key(make_client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "secret")
    c = make_client()
    assert c.get("/healthz").status_code == 200 and c.get("/readyz").status_code == 200


# ── Error envelope ────────────────────────────────────────
def test_validation_error_uses_envelope_with_request_id(make_client):
    r = make_client().post("/api/v1/search", json={"q": "bag", "limit": 0}, headers={"X-Request-ID": "abc-123"})
    err = r.json()["error"]
    assert r.status_code == 422 and err["code"] == "invalid_request"
    assert err["request_id"] == "abc-123" == r.headers["X-Request-ID"]
    assert err["details"][0]["field"] == "limit"


def test_unsafe_request_id_is_replaced(make_client):
    r = make_client().get("/healthz", headers={"X-Request-ID": "bad id\nwith newline"})
    assert r.headers["X-Request-ID"] != "bad id\nwith newline" and len(r.headers["X-Request-ID"]) == 32


def test_unknown_route_uses_envelope(make_client):
    r = make_client().get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/v1/search", {"q": "rug"}), ("post", "/api/v1/ask", {"q": "rug"}),
    ("get", "/api/v1/facets", None),
])
def test_search_outage_is_503_not_500(make_client, method, path, body):
    c = make_client(search=DownSearch())
    r = c.post(path, json=body) if method == "post" else c.get(path)
    err = r.json()["error"]
    assert r.status_code == 503 and err["code"] == "search_unavailable"
    assert "qdrant" not in err["message"].lower()  # no internals leak


# ── API keys ──────────────────────────────────────────────
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_missing_or_wrong_key_is_rejected(make_client, monkeypatch, headers):
    monkeypatch.setattr(settings, "API_KEYS", "k1, k2")
    r = make_client().post("/api/v1/search", json={"q": "rug"}, headers=headers)
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_any_configured_key_is_accepted(make_client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "k1, k2")
    c = make_client()
    for key in ("k1", "k2"):
        assert c.post("/api/v1/search", json={"q": "rug"}, headers={"X-API-Key": key}).status_code == 200


# ── Rate limits ───────────────────────────────────────────
def test_rate_limit_returns_429_with_retry_after(make_client, monkeypatch):
    monkeypatch.setattr(settings, "ASK_RATE_LIMIT_PER_MINUTE", 2)
    c = make_client()
    codes = [c.post("/api/v1/ask", json={"q": "rug"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/api/v1/ask", json={"q": "rug"})
    assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
    assert c.post("/api/v1/search", json={"q": "rug"}).status_code == 200  # separate, unlimited scope


def test_rate_limit_is_per_client(make_client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(settings, "API_KEYS", "a,b")
    c = make_client()
    assert c.post("/api/v1/search", json={"q": "rug"}, headers={"X-API-Key": "a"}).status_code == 200
    assert c.post("/api/v1/search", json={"q": "rug"}, headers={"X-API-Key": "b"}).status_code == 200
    assert c.post("/api/v1/search", json={"q": "rug"}, headers={"X-API-Key": "a"}).status_code == 429


def test_token_bucket_refills_over_time():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert [limiter.check("s", "c", 60) for _ in range(60)] == [None] * 60  # a full minute's burst
    assert limiter.check("s", "c", 60) == pytest.approx(1.0)  # empty: next token in 1 s
    now[0] += 1.0
    assert limiter.check("s", "c", 60) is None
    assert limiter.check("s", "c", 0) is None  # 0 = off


# ── Input guards ──────────────────────────────────────────
def test_control_characters_are_stripped(make_client):
    search = FakeSearch()
    c = make_client(search=search)
    assert c.post("/api/v1/search", json={"q": "grey\x00 rug​‮"}).status_code == 200
    assert search.calls[-1]["query"] == "grey rug"


def test_query_of_only_control_characters_is_rejected(make_client):
    assert make_client().post("/api/v1/search", json={"q": "\x00​"}).status_code == 422
