import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client import AsyncQdrantClient

from src.backend.services.search import ProductSearchService, rrf_fuse
from tests.conftest import COLLECTION


def _p(pid):
    return SimpleNamespace(id=pid, payload={"title": str(pid)})


def test_rrf_fuse_formula_and_order():
    dense = [_p("a"), _p("b"), _p("c")]
    sparse = [_p("b"), _p("d")]
    fused = rrf_fuse([dense, sparse], k=60, limit=10)
    scores = {pid: s for pid, s, _ in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 dense + rank 1 sparse
    assert scores["a"] == pytest.approx(1 / 61)
    assert [pid for pid, _, _ in fused][:2] == ["b", "a"]


def test_rrf_ties_broken_by_id_deterministically():
    # "z" and "a" both appear once at rank 1 → exactly equal scores.
    fused = rrf_fuse([[_p("z")], [_p("a")]], k=60, limit=2)
    assert [pid for pid, _, _ in fused] == ["a", "z"]
    fused_swapped = rrf_fuse([[_p("a")], [_p("z")]], k=60, limit=2)
    assert [pid for pid, _, _ in fused_swapped] == ["a", "z"]


def test_rrf_k_changes_consensus_vs_top_rank():
    # A: rank 1 in one list only. B: rank 4 in both lists.
    l1 = [_p("A"), _p("x1"), _p("x2"), _p("B")]
    l2 = [_p("y1"), _p("y2"), _p("y3"), _p("B")]
    top_k60 = rrf_fuse([l1, l2], k=60, limit=1)[0][0]
    assert top_k60 == "B"  # large k rewards appearing in both lists


@pytest.fixture(scope="module")
def services(indexed_path, embedder):
    client = AsyncQdrantClient(path=indexed_path)
    native = ProductSearchService(client=client, embedder=embedder, collection_name=COLLECTION,
                                  rrf_k=60, use_native_fusion=True)
    python = ProductSearchService(client=client, embedder=embedder, collection_name=COLLECTION,
                                  rrf_k=60, use_native_fusion=False)
    yield native, python
    asyncio.run(client.close())


def test_native_and_python_fusion_rank_identically(services, ground_truth):
    native, python = services
    for case in ground_truth:
        a = asyncio.run(native.search(case["query"], limit=5))
        b = asyncio.run(python.search(case["query"], limit=5))
        assert [r["id"] for r in a] == [r["id"] for r in b], case["query"]


def test_search_returns_catalog_ids_and_respects_limit(services):
    native, _ = services
    results = asyncio.run(native.search("headlamp rechargeable for camping", limit=3))
    assert len(results) == 3
    assert results[0]["id"] == "136"
    assert {"title", "price", "category", "score"} <= results[0].keys()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_modes(services, mode):
    native, _ = services
    results = asyncio.run(native.search("camping headlamp", limit=3, mode=mode))
    assert results and results[0]["id"] == "136"


def test_hybrid_can_return_more_than_prefetch_limit(indexed_path, embedder, services):
    native, _ = services
    small_prefetch = ProductSearchService(client=native.client, embedder=embedder,
                                          collection_name=COLLECTION, prefetch_limit=5)
    assert len(asyncio.run(small_prefetch.search("outdoor gear", limit=12))) == 12


def test_unknown_mode_rejected(services):
    with pytest.raises(ValueError):
        asyncio.run(services[0].search("x", mode="bogus"))


def test_query_uses_query_encoder(embedder):
    # BM25 query vectors weight every token 1.0 (IDF is applied by Qdrant).
    _, sparse = embedder.embed_query("waterproof hiking backpack")
    assert sparse["values"] and all(v == 1.0 for v in sparse["values"])
