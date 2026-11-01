import asyncio
import time

import pytest
from qdrant_client import AsyncQdrantClient

from src.backend.services.rerank import RerankerService
from src.backend.services.search import ProductSearchService
from src.ingestion.schemas import ProductRecord, build_search_text
from tests.conftest import COLLECTION


class FakeCrossEncoder:
    """Scores by how many query words appear in the text."""

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay, self.fail, self.seen = delay, fail, []

    def rerank(self, query, texts):
        self.seen.append(list(texts))
        if self.fail:
            raise RuntimeError("model crashed")
        time.sleep(self.delay)
        words = query.lower().split()
        return [sum(w in t.lower() for w in words) for t in texts]


def _results(*titles):
    return [{"id": str(i), "title": t, "score": 1 / (60 + i)} for i, t in enumerate(titles)]


def _reranker(**kw):
    model = kw.pop("model", FakeCrossEncoder())
    return RerankerService("fake", model=model, **kw), model


def test_rerank_reorders_head_and_keeps_tail():
    svc, _ = _reranker(candidates=3)
    results = _results("blue chair", "red oak table", "oak dining table", "tail item")
    out = asyncio.run(svc.rerank("oak table", results))
    assert out.reranked
    assert [r["title"] for r in out.results[:3]] == ["red oak table", "oak dining table", "blue chair"]
    assert out.results[3]["title"] == "tail item" and "rerank_score" not in out.results[3]
    assert out.results[0]["score"] == results[1]["score"]  # RRF score preserved


def test_rerank_timeout_keeps_original_order():
    svc, _ = _reranker(model=FakeCrossEncoder(delay=0.5), timeout_seconds=0.05)
    results = _results("a", "b", "c")
    out = asyncio.run(svc.rerank("b", results))
    assert not out.reranked and out.results == results


def test_rerank_error_keeps_original_order():
    svc, _ = _reranker(model=FakeCrossEncoder(fail=True))
    results = _results("a", "b")
    out = asyncio.run(svc.rerank("b", results))
    assert not out.reranked and out.results == results


def test_rerank_skips_single_result():
    svc, model = _reranker()
    out = asyncio.run(svc.rerank("x", _results("only")))
    assert not out.reranked and model.seen == []


def test_reranker_scores_the_indexed_text():
    record = ProductRecord(id="1", title="Oak Table", price=10, product_class="Tables",
                           category_path=["Furniture", "Tables"], features={"color": "brown"},
                           description="Solid oak.")
    payload = {"product_id": "1", **record.model_dump(exclude={"id"}), "score": 0.1, "id": "1"}
    assert build_search_text(payload) == record.get_text_payload()


# ── Search integration (local Qdrant, real embeddings, fake cross-encoder) ──
@pytest.fixture(scope="module")
def client(indexed_path):
    c = AsyncQdrantClient(path=indexed_path)
    yield c
    asyncio.run(c.close())


def test_hybrid_rerank_mode(client, embedder):
    reranker, model = _reranker(candidates=5)
    svc = ProductSearchService(client=client, embedder=embedder, collection_name=COLLECTION,
                               reranker=reranker)
    results = asyncio.run(svc.search("camping headlamp", limit=3))  # default mode = hybrid_rerank
    assert len(results) == 3 and "rerank_score" in results[0]
    assert len(model.seen[0]) == 5  # fetched `candidates` results, reranked, cut to limit
    plain = asyncio.run(svc.search("camping headlamp", limit=3, mode="hybrid"))
    assert "rerank_score" not in plain[0]


def test_hybrid_rerank_without_reranker_rejected(client, embedder):
    svc = ProductSearchService(client=client, embedder=embedder, collection_name=COLLECTION)
    with pytest.raises(ValueError):
        asyncio.run(svc.search("x", mode="hybrid_rerank"))
