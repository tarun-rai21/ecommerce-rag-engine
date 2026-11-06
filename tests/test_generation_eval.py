"""Judge parsing, deterministic answer checks, caching and agreement statistics."""
import csv
import json
from types import SimpleNamespace

import pytest

from src.backend.services.llm import REFUSAL_MESSAGE, build_messages, clean_output
from src.backend.api.v1.endpoints import build_context
from src.evaluation.generation_eval import calibrate, cohen_kappa, export_labels, spearman, summarize
from src.evaluation.judge import (
    CachedLLM, dollar_amounts, is_refusal, mentions_context, overclaims_catalog, parse_judgment, ungrounded_prices,
)


@pytest.mark.parametrize("text,expected", [
    ("These are the only products in the provided context with turquoise.", True),
    ("Based on the data supplied, ...", True),
    ("Nothing in the product data mentions size.", True),
    ("Here are three turquoise pillows.", False),
])
def test_mentions_context(text, expected):
    assert mentions_context(text) is expected


# ── Judge output parsing ──────────────────────────────────
def test_parse_judgment_scores_claims():
    j = parse_judgment(json.dumps({"claims": [{"claim": "a", "supported": True},
                                              {"claim": "b", "supported": False}], "relevance": 4}))
    assert j.faithfulness == 0.5 and j.fully_faithful is False and j.relevance == 4 and j.parse_error is None


def test_parse_judgment_tolerates_fences_and_think_blocks():
    text = '<think>hmm {"x": 1}</think>```json\n{"claims": [{"claim": "a", "supported": true}], "relevance": "5"}\n```'
    j = parse_judgment(text)
    assert j.faithfulness == 1.0 and j.relevance == 5


@pytest.mark.parametrize("text", ["no json here", "{bad json}"])
def test_parse_judgment_reports_errors(text):
    assert parse_judgment(text).parse_error


def test_parse_judgment_drops_malformed_claims_and_bad_relevance():
    j = parse_judgment(json.dumps({"claims": [{"claim": "a", "supported": "yes"}, "junk"], "relevance": 9}))
    assert j.claims == [] and j.faithfulness is None and j.relevance is None


# ── Deterministic checks ──────────────────────────────────
def test_is_refusal():
    assert is_refusal(REFUSAL_MESSAGE)
    assert is_refusal("Sorry. I cannot  find a matching product for that.")
    assert not is_refusal("Here are two oak coffee tables.")


@pytest.mark.parametrize("text,expected", [
    ("These are the only products in the catalog that match.", True),
    ("They are the only ones that feature turquoise.", True),
    ("Here are all the king beds available in our catalog:", True),
    ("Here are three oak coffee tables.", False),
    ("Only $45, the cheapest option here.", False),
])
def test_overclaims_catalog(text, expected):
    assert overclaims_catalog(text) is expected


def test_dollar_amounts():
    assert dollar_amounts("$1,299.99 or $ 45 and $3.5") == [1299.99, 45.0, 3.5]


def test_ungrounded_prices_accepts_exact_and_rounded():
    ctx = "Price: $189.99 | Price: $45.00\n(filters: price at most $300.00)"
    assert ungrounded_prices("costs $189.99, about $190, under $300, $45", ctx) == []
    assert ungrounded_prices("only $120", ctx) == [120.0]


# ── Production prompt reuse ───────────────────────────────
def test_build_messages_matches_production_layout():
    msgs = build_messages("rug", "- Title: x")
    assert msgs[0]["role"] == "system" and REFUSAL_MESSAGE in msgs[0]["content"]
    assert msgs[1]["content"] == "VERIFIED PRODUCT CONTEXT:\n- Title: x\n\nUSER QUESTION:\nrug"
    assert clean_output("<think>x</think> answer ") == "answer"


# ── Cache ─────────────────────────────────────────────────
class _FakeClient:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"reply {self.calls}"))])


def test_cached_llm_reuses_identical_requests(tmp_path):
    client = _FakeClient()
    llm = CachedLLM(client, cache_path=tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    assert llm.complete("m", msgs, temperature=0) == "reply 1"
    assert llm.complete("m", msgs, temperature=0) == "reply 1"
    assert llm.complete("m", msgs, temperature=0.5) == "reply 2"  # params are part of the key
    assert client.calls == 2 and llm.hits == 1
    llm.close()


# ── Stable prompts and report-based calibration ───────────
def test_build_context_is_independent_of_attribute_order():
    base = {"title": "t", "price": 1.0, "description": "d"}
    a = build_context([{**base, "features": {"style": "modern", "color": "grey"}}])
    b = build_context([{**base, "features": {"color": "grey", "style": "modern"}}])
    assert a == b and "Attributes: color: grey, style: modern" in a


def _answered(item_id, faithful=True, relevance=5):
    return {"item_id": item_id, "segment": "plain", "query": "q", "context": "c", "answer": "a", "refused": False,
            "fully_faithful": faithful, "relevance": relevance}


def test_export_is_blind_and_calibrate_compares_with_report(tmp_path):
    rows = [_answered("plain:1"), _answered("plain:2", faithful=False, relevance=2),
            {**_answered("plain:3"), "refused": True}]
    sheet = tmp_path / "labels.csv"
    export_labels(rows, sheet)
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        labelled = list(csv.DictReader(fh))
    assert {r["item_id"] for r in labelled} == {"plain:1", "plain:2"}  # refusals excluded
    assert "fully_faithful" not in labelled[0] and "relevance" not in labelled[0]  # judge verdicts hidden

    verdicts = {"plain:1": ("1", "5"), "plain:2": ("0", "3")}
    for r in labelled:
        r["reviewer_fully_faithful"], r["reviewer_relevance"] = verdicts[r["item_id"]]
    with sheet.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=labelled[0].keys())
        w.writeheader()
        w.writerows(labelled)
    result = calibrate(rows, sheet)
    assert "2 independently labelled" in result and "agreement 100%" in result and "within 1 point 50%" not in result


# ── Agreement statistics ──────────────────────────────────
def test_cohen_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_spearman_handles_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1.0)


def test_summarize_refusal_accounting():
    def row(seg, refused, exact=False, faith=None):
        return {"segment": seg, "refused": refused, "exact_in_context": exact, "faithfulness": faith,
                "fully_faithful": None if faith is None else faith == 1.0, "relevance": None if refused else 5,
                "claims": [] if faith is None else [{"supported": True}], "ungrounded_prices": [], "judge_error": None,
                "answer": "x"}
    rows = [row("ooc", True), row("ooc", False, faith=1.0), row("plain", False, True, 1.0), row("plain", True, True)]
    s = summarize(rows)
    assert s["refusal"]["ooc_correct_refusal"]["mean"] == 0.5
    assert s["refusal"]["in_catalog_false_refusal"]["mean"] == 0.5
    assert s["refusal"]["false_refusal_when_exact_retrieved"]["n"] == 2
