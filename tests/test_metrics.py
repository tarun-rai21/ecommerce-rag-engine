"""/metrics: what gets counted, and its optional token."""
import json
import re

import pytest
from loguru import logger

from config.settings import settings
from src.backend.api.security import rate_limiter
from src.backend.logging_config import configure_logging
from tests.fakes import FakeLLM, FakeRedis


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name, value in (("API_KEYS", ""), ("RATE_LIMIT_PER_MINUTE", 0), ("ASK_RATE_LIMIT_PER_MINUTE", 0),
                        ("METRICS_TOKEN", "")):
        monkeypatch.setattr(settings, name, value)
    rate_limiter.reset()


def metric(client, name: str, **labels) -> float:
    """Current value of one sample in /metrics (0 if absent)."""
    text = client.get("/metrics").text
    want = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    for line in text.splitlines():
        m = re.match(rf"^{name}(?:{{(.*)}})? (\S+)$", line)
        if m and ",".join(sorted((m.group(1) or "").split(","))) == ",".join(sorted(want.split(","))):
            return float(m.group(2))
    return 0.0


def test_ask_counts_answers_tokens_cache_and_stages(make_client):
    c = make_client(llm=FakeLLM(), redis=FakeRedis())
    before = {
        "llm": metric(c, "rag_answers_total", source="llm"),
        "cache": metric(c, "rag_answers_total", source="cache"),
        "prompt": metric(c, "rag_llm_tokens_total", type="prompt"),
        "miss": metric(c, "rag_cache_lookups_total", result="miss"),
        "hit": metric(c, "rag_cache_lookups_total", result="hit"),
        "llm_stage": metric(c, "rag_stage_duration_seconds_count", stage="llm"),
    }
    c.post("/api/v1/ask", json={"q": "bag"})
    c.post("/api/v1/ask", json={"q": "bag"})  # cache hit
    assert metric(c, "rag_answers_total", source="llm") == before["llm"] + 1
    assert metric(c, "rag_answers_total", source="cache") == before["cache"] + 1
    assert metric(c, "rag_llm_tokens_total", type="prompt") == before["prompt"] + 900  # FakeLLM's usage
    assert metric(c, "rag_cache_lookups_total", result="miss") == before["miss"] + 1
    assert metric(c, "rag_cache_lookups_total", result="hit") == before["hit"] + 1
    assert metric(c, "rag_stage_duration_seconds_count", stage="llm") == before["llm_stage"] + 1


def test_requests_are_labelled_by_route_template_not_raw_path(make_client):
    c = make_client()
    before = metric(c, "rag_requests_total", route="unmatched", method="GET", status="404")
    c.get("/some/random/path/123")
    c.get("/another/one")
    assert metric(c, "rag_requests_total", route="unmatched", method="GET", status="404") == before + 2
    c.post("/api/v1/search", json={"q": "rug"})
    assert metric(c, "rag_requests_total", route="/api/v1/search", method="POST", status="200") >= 1
    assert "/some/random/path/123" not in c.get("/metrics").text


def test_safeguard_refusals_are_counted(make_client, monkeypatch):
    c = make_client()
    auth, limited = metric(c, "rag_auth_failures_total"), metric(c, "rag_rate_limited_total", scope="ask")
    monkeypatch.setattr(settings, "API_KEYS", "k")
    c.post("/api/v1/search", json={"q": "rug"})
    monkeypatch.setattr(settings, "ASK_RATE_LIMIT_PER_MINUTE", 1)
    for _ in range(2):
        c.post("/api/v1/ask", json={"q": "rug"}, headers={"X-API-Key": "k"})
    assert metric(c, "rag_auth_failures_total") == auth + 1
    assert metric(c, "rag_rate_limited_total", scope="ask") == limited + 1


def test_cache_breaker_and_errors_show_up(make_client):
    c = make_client(redis=FakeRedis(fail=True))
    errors = metric(c, "rag_cache_lookups_total", result="error")
    bypassed = metric(c, "rag_cache_lookups_total", result="bypassed")
    c.post("/api/v1/ask", json={"q": "bag"})   # connection error opens the breaker
    c.post("/api/v1/ask", json={"q": "pen"})   # skipped while it's open
    assert metric(c, "rag_cache_lookups_total", result="error") == errors + 1
    assert metric(c, "rag_cache_lookups_total", result="bypassed") == bypassed + 1
    assert metric(c, "rag_cache_breaker_open") == 1.0


def test_metrics_token(make_client, monkeypatch):
    c = make_client()
    monkeypatch.setattr(settings, "METRICS_TOKEN", "t0k")
    assert c.get("/metrics").status_code == 401
    assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = c.get("/metrics", headers={"Authorization": "Bearer t0k"})
    assert r.status_code == 200 and "rag_requests_total" in r.text


def test_metrics_not_in_api_docs(make_client):
    assert "/metrics" not in make_client().get("/openapi.json").json()["paths"]


def test_json_logs_carry_the_request_id(make_client, capsys):
    configure_logging("json")
    try:
        make_client().post("/api/v1/search", json={"q": "rug"}, headers={"X-Request-ID": "trace-42"})
        lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
        access = [rec for rec in lines if "/api/v1/search" in rec["record"]["message"]]
        assert access and access[-1]["record"]["extra"]["request_id"] == "trace-42"
    finally:
        configure_logging("text")
        logger.remove()


def test_known_series_exist_at_zero_before_any_traffic():
    from prometheus_client import generate_latest
    text = generate_latest().decode()
    for line in ('rag_answers_total{source="llm_error"}', 'rag_cache_lookups_total{result="bypassed"}',
                 'rag_llm_tokens_total{type="completion"}', 'rag_rate_limited_total{scope="search"}'):
        assert line in text
