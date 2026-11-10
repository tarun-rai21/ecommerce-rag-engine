"""Feedback: ratings joined to the recorded answer they refer to."""
import json

import pytest

from config.settings import settings
from src.backend.api.security import rate_limiter
from tests.fakes import FakeLLM, FakeRedis
from tests.test_streaming import StreamingLLM, events


@pytest.fixture(autouse=True)
def open_api(monkeypatch):
    for name, value in (("API_KEYS", ""), ("RATE_LIMIT_PER_MINUTE", 0), ("ASK_RATE_LIMIT_PER_MINUTE", 0)):
        monkeypatch.setattr(settings, name, value)
    rate_limiter.reset()


def test_rating_is_stored_with_what_the_shopper_saw(make_client):
    redis = FakeRedis()
    c = make_client(redis=redis, llm=FakeLLM())
    r = c.post("/api/v1/ask", json={"q": "grey rug under $300", "limit": 2}, headers={"X-Request-ID": "req-1"})
    assert r.status_code == 200
    fb = c.post("/api/v1/feedback", json={"request_id": "req-1", "rating": "down", "comment": "too pricey"})
    assert fb.status_code == 201 and fb.json() == {"status": "recorded"}
    entry = json.loads(redis.store["rag:feedback"][0])
    assert entry["rating"] == "down" and entry["comment"] == "too pricey"
    seen = entry["interaction"]
    assert seen["query"] == "grey rug under $300" and seen["interpreted_query"] == "grey rug"
    assert seen["filters"] == {"max_price": 300.0} and seen["retrieved_ids"] == ["100", "101"]
    assert seen["answer"] == "answer" and seen["source"] == "llm" and seen["model"] == settings.LLM_MODEL


def test_streamed_and_cached_answers_can_be_rated(make_client):
    c = make_client(redis=FakeRedis(), llm=StreamingLLM())
    c.post("/api/v1/ask/stream", json={"q": "rug"}, headers={"X-Request-ID": "s-1"})
    meta = events(c.post("/api/v1/ask/stream", json={"q": "rug"}, headers={"X-Request-ID": "s-2"}))[0][1]
    assert meta["source"] == "cache" and meta["request_id"] == "s-2"
    for rid in ("s-1", "s-2"):
        assert c.post("/api/v1/feedback", json={"request_id": rid, "rating": "up"}).status_code == 201


def test_one_rating_per_answer(make_client):
    c = make_client(redis=FakeRedis())
    c.post("/api/v1/ask", json={"q": "rug"}, headers={"X-Request-ID": "r1"})
    assert c.post("/api/v1/feedback", json={"request_id": "r1", "rating": "up"}).status_code == 201
    r = c.post("/api/v1/feedback", json={"request_id": "r1", "rating": "down"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_rated"


def test_unknown_request_id_is_404(make_client):
    r = make_client().post("/api/v1/feedback", json={"request_id": "never-answered", "rating": "up"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_request"


@pytest.mark.parametrize("body", [
    {"request_id": "r1", "rating": "meh"},
    {"request_id": "bad id with spaces", "rating": "up"},
    {"request_id": "r1", "rating": "up", "comment": "x" * 501},
])
def test_invalid_feedback_rejected(make_client, body):
    assert make_client().post("/api/v1/feedback", json=body).status_code == 422


def test_redis_down_answers_still_work_but_feedback_is_503(make_client):
    c = make_client(redis=FakeRedis(fail=True))
    assert c.post("/api/v1/ask", json={"q": "rug"}, headers={"X-Request-ID": "r9"}).status_code == 200
    r = c.post("/api/v1/feedback", json={"request_id": "r9", "rating": "up"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "feedback_unavailable"


def test_feedback_needs_the_api_key(make_client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "k")
    assert make_client().post("/api/v1/feedback", json={"request_id": "r1", "rating": "up"}).status_code == 401


def test_recording_is_skipped_while_the_cache_breaker_is_open():
    import asyncio

    from src.backend.services.feedback import FeedbackStore
    from tests.fakes import FakeRedis

    redis = FakeRedis()
    store = FeedbackStore(redis, available=lambda: False)
    asyncio.run(store.record_interaction("r-1", {"query": "rug"}))
    assert redis.store == {}  # Redis wasn't touched
