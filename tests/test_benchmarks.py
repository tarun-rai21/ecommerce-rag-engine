import json

import pytest

from benchmarks.cost import answer_costs, report


def test_answer_cost_uses_input_and_output_prices():
    costs = answer_costs([{"prompt_tokens": 1_000_000, "completion_tokens": 0},
                          {"prompt_tokens": 1000, "completion_tokens": 200}], {"input": 0.075, "output": 0.30})
    assert costs[0] == pytest.approx(0.075)
    assert costs[1] == pytest.approx((1000 * 0.075 + 200 * 0.30) / 1e6)


def test_report_counts_only_llm_answers_and_scales_by_hit_rate(tmp_path):
    usage = {"prompt_tokens": 1000, "completion_tokens": 200}
    rows = [{"source": "llm", "usage": usage}, {"source": "llm", "usage": usage},
            {"source": "llm_error", "usage": None}, {"source": "cache", "usage": usage}]
    path = tmp_path / "load_x.json"
    path.write_text(json.dumps({"ask-miss": rows}), encoding="utf-8")
    out = report(path, "openai/gpt-oss-20b")
    assert "from 2 uncached answers" in out
    assert "| 0% | $0.135 [$0.135, $0.135] |" in out   # (1000 x 0.075 + 200 x 0.30) / 1e6 x 1000
    assert "| 50% | $0.068 [$0.068, $0.068] |" in out
