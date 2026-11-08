"""Cost per query of /ask, from the token usage recorded by the load test.

    python -m benchmarks.cost                       # uses the latest reports/load_*.json
    python -m benchmarks.cost --report reports/load_<...>.json

Only uncached /ask answers call the LLM. Embedding and search run on our own CPU and
add no per-query fee, and a cache hit costs nothing, so cost per 1,000 questions is
(1 - cache hit rate) x 1,000 x cost per LLM answer. The hit rate depends on real
traffic, which this project doesn't have, so the cost is shown for several rates.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Groq on-demand prices, USD per 1M tokens, read on 2026-11-09 from
# https://console.groq.com/docs/models ("PRICE PER 1M TOKENS"). Update when they change.
PRICES = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}
HIT_RATES = (0.0, 0.25, 0.5, 0.75)
SEED = 11


def latest_load_report() -> Path:
    reports = sorted(REPORTS_DIR.glob("load_*.json"), key=lambda p: p.stat().st_mtime)
    with_usage = [p for p in reports if json.loads(p.read_text(encoding="utf-8")).get("ask-miss")]
    if not with_usage:
        raise SystemExit("No load report with an ask-miss part: run python -m benchmarks.load_test first.")
    return with_usage[-1]


def answer_costs(usage: list[dict], price: dict) -> np.ndarray:
    """USD per LLM answer. completion_tokens includes reasoning tokens, which are billed as output."""
    return np.array([(u["prompt_tokens"] * price["input"] + u["completion_tokens"] * price["output"]) / 1e6
                     for u in usage])


def mean_ci(x: np.ndarray, n: int = 5000) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    boot = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def report(path: Path, model: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    usage = [r["usage"] for r in data["ask-miss"] if r.get("source") == "llm" and r.get("usage")]
    if not usage:
        raise SystemExit(f"{path.name} has no token usage for LLM answers.")
    price = PRICES[model]
    costs = answer_costs(usage, price)
    prompt = np.array([u["prompt_tokens"] for u in usage])
    completion = np.array([u["completion_tokens"] for u in usage])
    per_1k = mean_ci(costs * 1000)
    input_share = float((prompt * price["input"]).sum() / (costs * 1e6).sum())

    lines = [
        f"Cost of `/ask` with `{model}` at ${price['input']}/1M input and ${price['output']}/1M output tokens "
        f"(Groq, read 2026-11-09). Token usage from {len(usage)} uncached answers in `{path.name}`; 95% bootstrap CIs.",
        "",
        "| | Mean | Range |",
        "|---|---|---|",
        f"| Prompt tokens per answer | {prompt.mean():.0f} | {prompt.min()}–{prompt.max()} |",
        f"| Completion tokens per answer (incl. reasoning) | {completion.mean():.0f} | {completion.min()}–{completion.max()} |",
        f"| Cost per LLM answer | ${costs.mean():.6f} | ${costs.min():.6f}–${costs.max():.6f} |",
        "",
        f"Input is {input_share:.0%} of the cost.",
        "",
        "| Cache hit rate | Cost per 1,000 questions |",
        "|---|---|",
    ]
    for rate in HIT_RATES:
        m, lo, hi = (v * (1 - rate) for v in per_1k)
        lines.append(f"| {rate:.0%} | ${m:.3f} [${lo:.3f}, ${hi:.3f}] |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, help="load-test report JSON (default: latest with an ask-miss part)")
    p.add_argument("--model", default="openai/gpt-oss-20b", choices=sorted(PRICES))
    args = p.parse_args(argv)
    print("\n" + report(args.report or latest_load_report(), args.model) + "\n")


if __name__ == "__main__":
    main()
