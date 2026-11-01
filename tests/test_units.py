import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService, UNAVAILABLE_MESSAGE
from src.evaluation import metrics as m


# ── Cache keys ────────────────────────────────────────────
def test_cache_key_normalises_query_and_includes_params():
    k1 = CacheService.make_key("ask", "Waterproof  Bag ", limit=5)
    k2 = CacheService.make_key("ask", "waterproof bag", limit=5)
    k3 = CacheService.make_key("ask", "waterproof bag", limit=10)
    assert k1 == k2 and k1 != k3
    assert k1.startswith("rag:") and "limit=5" in k1


class _SlowOrBrokenRedis:
    def __init__(self, mode):
        self.mode, self.calls = mode, 0

    async def get(self, key):
        self.calls += 1
        if self.mode == "slow":
            await asyncio.sleep(5)
        raise OSError("Name or service not known")


def test_cache_op_is_time_bounded():
    cache = CacheService(client=_SlowOrBrokenRedis("slow"), op_timeout=0.05, cooldown=30)
    start = time.perf_counter()
    assert asyncio.run(cache.get("k")) is None
    assert time.perf_counter() - start < 1.0


def test_cache_circuit_breaker_skips_redis_after_failure():
    redis = _SlowOrBrokenRedis("broken")
    cache = CacheService(client=redis, op_timeout=0.5, cooldown=30)
    assert asyncio.run(cache.get("k")) is None
    assert asyncio.run(cache.get("k")) is None
    assert redis.calls == 1 and not cache.available  # second call never touched Redis


def test_cache_recovers_after_cooldown():
    redis = _SlowOrBrokenRedis("broken")
    cache = CacheService(client=redis, op_timeout=0.5, cooldown=0)
    asyncio.run(cache.get("k"))
    asyncio.run(cache.get("k"))
    assert redis.calls == 2  # cooldown 0 → retried


# ── Metrics ───────────────────────────────────────────────
def test_ranking_metrics():
    retrieved, relevant = ["a", "x", "b", "y", "z"], {"a", "b", "c"}
    assert m.hit_rate_at_k(retrieved, relevant, 5) == 1.0
    assert m.precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)
    assert m.max_precision_at_k(relevant, 5) == pytest.approx(3 / 5)
    assert m.recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert m.reciprocal_rank(["x", "a"], relevant) == 0.5
    assert m.reciprocal_rank(["x", "y"], relevant) == 0.0


def test_ndcg():
    gains = {"a": 1.0, "b": 1.0}
    assert m.ndcg_at_k(["a", "b", "x"], gains, 3) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))  # only 'b' at rank 2 of an ideal 2
    assert m.ndcg_at_k(["x", "b"], gains, 2) == pytest.approx(expected)
    assert m.ndcg_at_k(["x"], {}, 1) == 0.0


def test_keyword_coverage():
    assert m.keyword_coverage("The Rugged Backpack costs $45.99", ["backpack", "45.99", "tent"]) == pytest.approx(2 / 3)
    assert m.keyword_coverage("anything", []) == 0.0


# ── LLM service ───────────────────────────────────────────
def _fake_groq(content):
    create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_llm_success():
    result = asyncio.run(LLMService(client=_fake_groq(" hello ")).generate_answer("q", "ctx"))
    assert result.ok and result.answer == "hello"


def test_llm_strips_think_block():
    result = asyncio.run(LLMService(client=_fake_groq("<think>plan</think>\n Answer")).generate_answer("q", "c"))
    assert result.ok and result.answer == "Answer"


@pytest.mark.parametrize("content", [None, "   ", "<think>only thinking</think>"])
def test_llm_empty_content_is_a_failure(content):
    result = asyncio.run(LLMService(client=_fake_groq(content)).generate_answer("q", "ctx"))
    assert not result.ok and result.answer == UNAVAILABLE_MESSAGE
