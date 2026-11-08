"""WANDS loader, synthetic pricing, embedding cache, labelled queries, statistics."""
import math

import numpy as np
import pytest

from src.evaluation.datasets import EvalQuery, load_wands_queries, split_of
from src.evaluation.retrieval_eval import score_query
from src.evaluation.stats import bootstrap_mean, paired_difference
from src.ingestion.embedding_cache import EmbeddingCache
from src.ingestion.ingest import embed_with_cache
from src.ingestion.loaders.wands import iter_wands_products, parse_category_path, parse_features
from src.ingestion.pricing import CATEGORY_MEDIANS, synthetic_price
from src.ingestion.schemas import ProductRecord

PRODUCT_TSV = (
    "product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description\t"
    "product_features\trating_count\taverage_rating\treview_count\n"
    "7\tsolid wood platform bed\tBeds\tFurniture / Bedroom Furniture / Beds\tdeep sleep.\t"
    "color : caramel|overallwidth-sidetoside:64.7|primarymaterial : wood\t15.0\t4.5\t15.0\n"
    "8\tdinosaur wall art\t\t\t\tcolor:green\t\t\t\n"
)
QUERY_TSV = "query_id\tquery\tquery_class\n0\tsalon chair\tMassage Chairs\n1\tdinosaur\tKids Wall Décor\n"
LABEL_TSV = (
    "id\tquery_id\tproduct_id\tlabel\n"
    "0\t0\t7\tPartial\n1\t0\t7\tExact\n"   # conflicting duplicate → keep max (Exact)
    "2\t0\t8\tIrrelevant\n3\t1\t8\tPartial\n"
)


@pytest.fixture
def wands_dir(tmp_path):
    (tmp_path / "product.csv").write_text(PRODUCT_TSV, encoding="utf-8")
    (tmp_path / "query.csv").write_text(QUERY_TSV, encoding="utf-8")
    (tmp_path / "label.csv").write_text(LABEL_TSV, encoding="utf-8")
    return tmp_path


# ── Loader ────────────────────────────────────────────────
def test_parse_features_keeps_selected_keys():
    assert parse_features("color : grey|overallwidth-sidetoside:64.7|style:modern|bogus") == {
        "color": "grey", "style": "modern"}
    assert parse_features(None) == {}


def test_parse_category_path():
    assert parse_category_path("Furniture / Bedroom Furniture / Beds") == ["Furniture", "Bedroom Furniture", "Beds"]
    assert parse_category_path(None) == []


def test_wands_loader_produces_valid_records(wands_dir):
    raw = list(iter_wands_products(wands_dir))
    records = [ProductRecord(**r) for r in raw]
    bed, art = records
    assert bed.category == "Furniture" and bed.product_class == "Beds"
    assert bed.features == {"color": "caramel", "primarymaterial": "wood"}
    assert bed.rating == 4.5 and bed.review_count == 15 and bed.price_is_synthetic
    assert art.category == "Unknown" and art.product_class is None and art.rating is None
    text = bed.get_text_payload()
    assert text.startswith("solid wood platform bed. Beds. Furniture > Bedroom Furniture > Beds")
    assert "color: caramel" in text and "Title:" not in text


# ── Synthetic pricing ─────────────────────────────────────
def test_synthetic_price_is_deterministic_and_retail_shaped():
    a, b = synthetic_price("123", "Furniture"), synthetic_price("123", "Furniture")
    assert a == b and a > 0 and round(a % 1, 2) == 0.99
    assert synthetic_price("123", "Furniture", seed=1) != a


def test_synthetic_price_medians_track_category():
    for category, median in [("Furniture", CATEGORY_MEDIANS["Furniture"]), ("Kitchen & Tabletop", 40.0)]:
        prices = sorted(synthetic_price(str(i), category) for i in range(2000))
        assert abs(prices[1000] - median) / median < 0.1


# ── Embedding cache ───────────────────────────────────────
class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return ([[float(len(t))] * 3 for t in texts],
                [{"indices": [len(t)], "values": [1.0]} for t in texts])


def test_embedding_cache_roundtrip_and_reuse(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite", model_tag="m")
    emb = CountingEmbedder()
    d1, s1, new1 = embed_with_cache(emb, ["ab", "abc", "ab"], cache)
    assert new1 == 2 and emb.calls == 2          # duplicate text embedded once
    d2, s2, new2 = embed_with_cache(emb, ["abc", "ab"], cache)
    assert new2 == 0 and emb.calls == 2          # all from cache
    assert d2 == [d1[1], d1[0]] and s2 == [s1[1], s1[0]]
    assert len(cache) == 2
    cache.close()
    reopened = EmbeddingCache(tmp_path / "e.sqlite", model_tag="m")
    assert len(reopened) == 2                    # survives restart (resumable ingestion)
    assert len(EmbeddingCache(tmp_path / "e.sqlite", model_tag="other")) == 0


# ── Labelled queries ──────────────────────────────────────
def test_load_queries_keeps_max_gain_on_conflicts(wands_dir):
    queries = {q.query_id: q for q in load_wands_queries(wands_dir)}
    assert queries["0"].gains == {"7": 2.0, "8": 0.0}
    assert queries["0"].exact == {"7"}
    assert queries["1"].query_class == "Kids Wall Décor" and queries["1"].exact == set()


def test_split_is_deterministic_and_roughly_one_third_dev():
    splits = [split_of(str(i)) for i in range(3000)]
    assert splits == [split_of(str(i)) for i in range(3000)]
    assert 0.30 < splits.count("dev") / len(splits) < 0.37


def test_score_query():
    q = EvalQuery("0", "q", None, {"a": 2.0, "b": 1.0, "c": 0.0})
    scores = score_query(q, top=["x", "a", "c"], deep=["x", "a", "c", "b"])
    assert scores["mrr@10"] == 0.5 and scores["recall@100"] == 1.0
    assert scores["judged@10"] == pytest.approx(2 / 10)
    assert 0 < scores["ndcg@10"] < 1
    no_exact = score_query(EvalQuery("1", "q", None, {"b": 1.0}), top=["b"], deep=["b"])
    assert no_exact["recall@100"] is None and no_exact["ndcg@10"] == pytest.approx(1.0)


# ── Statistics ────────────────────────────────────────────
def test_bootstrap_ci_contains_mean_and_narrows_with_n():
    rng = np.random.default_rng(0)
    small, large = bootstrap_mean(list(rng.random(30))), bootstrap_mean(list(rng.random(3000)))
    assert small.low <= small.mean <= small.high
    assert (large.high - large.low) < (small.high - small.low)
    assert math.isnan(bootstrap_mean([]).mean)


def test_paired_difference_detects_consistent_gain():
    base = list(np.linspace(0, 1, 100))
    better = [b + 0.05 for b in base]
    assert paired_difference(better, base).low > 0
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 2.0])
