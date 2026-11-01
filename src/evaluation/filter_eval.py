"""Price-constrained retrieval benchmark built from WANDS.

WANDS queries never state a budget, so each query gets one: X = the median
synthetic price of its Exact products (so about half of them fit), and the query
becomes "<query> under $X". A product is relevant only if it is judged relevant
AND priced <= X.

Systems (all hybrid retrieval, production settings):
  A  no_parsing   search the raw text "<query> under $X", no filter (old behaviour)
  B  post_filter  parse, search the cleaned text, then drop over-budget results
                  from the top 10 (common shortcut; leaves gaps)
  C  pre_filter   parse, search the cleaned text with a Qdrant price filter applied
                  inside each prefetch (this system)

Metrics: nDCG@10 with constraint-aware gains, violation rate (share of returned
results over budget), fill rate (returned results / 10). Bootstrap 95% CIs.

Prices are synthetic (WANDS has none), so this measures filter mechanics, not
real-world price behaviour.

Usage: python -m src.evaluation.filter_eval --split test
"""
import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.backend.services.query_parser import parse_query
from src.backend.services.search import ProductSearchService
from src.evaluation.datasets import load_wands_queries
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.retrieval_eval import MemoEmbedder, _git_sha
from src.evaluation.stats import bootstrap_mean, paired_difference
from src.ingestion.embedder import ProductEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
K = 10
SYSTEMS = ("no_parsing", "post_filter", "pre_filter")


async def fetch_prices(service: ProductSearchService, product_ids: set[str]) -> dict[str, float]:
    ids = [int(p) for p in product_ids]
    prices: dict[str, float] = {}
    for i in range(0, len(ids), 500):
        points = await service.client.retrieve(service.collection_name, ids=ids[i:i + 500], with_payload=["price"])
        prices.update({str(p.id): float(p.payload["price"]) for p in points})
    return prices


def budget_for(exact_prices: list[float]) -> float:
    return float(math.floor(statistics.median(exact_prices)))


async def run(args: argparse.Namespace) -> None:
    queries = [q for q in load_wands_queries(split=args.split) if q.exact]
    service = ProductSearchService(embedder=MemoEmbedder(ProductEmbedder()), collection_name=args.collection)
    prices = await fetch_prices(service, {p for q in queries for p in q.exact})

    rows: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
    try:
        for i, q in enumerate(queries, start=1):
            budget = budget_for([prices[p] for p in q.exact if p in prices])
            constrained = f"{q.text} under ${budget:.0f}"
            parsed = parse_query(constrained)
            if parsed.filters.max_price != budget:
                raise AssertionError(f"parser missed the budget in {constrained!r}: {parsed}")
            gains = {p: g for p, g in q.gains.items() if p in prices and prices[p] <= budget}

            raw = await service.search(constrained, limit=K)
            clean = await service.search(parsed.text, limit=K)
            filtered = await service.search(parsed.text, limit=K, filters=parsed.filters)
            results = {
                "no_parsing": raw,
                "post_filter": [r for r in clean if r["price"] <= budget],
                "pre_filter": filtered,
            }
            for system, res in results.items():
                ids = [str(r["id"]) for r in res]
                rows[system].append({
                    "query_id": q.query_id, "query": constrained, "budget": budget,
                    "ndcg@10": ndcg_at_k(ids, gains, K),
                    "violation_rate": (sum(r["price"] > budget for r in res) / len(res)) if res else 0.0,
                    "fill_rate": len(res) / K,
                })
            if i % 50 == 0:
                logger.info(f"{i}/{len(queries)} queries")
    finally:
        await service.close()

    summary = {
        s: {m: vars(bootstrap_mean([r[m] for r in rows[s]])) for m in ("ndcg@10", "violation_rate", "fill_rate")}
        for s in SYSTEMS
    }
    for s in ("no_parsing", "post_filter"):
        summary["pre_filter"][f"ndcg_vs_{s}"] = vars(paired_difference(
            [r["ndcg@10"] for r in rows["pre_filter"]], [r["ndcg@10"] for r in rows[s]]))

    lines = [
        f"Split: **{args.split}** ({len(queries)} queries with ≥1 Exact; budget = median price of the query's Exact products).",
        "",
        "| System | nDCG@10 (budget-aware) | Over-budget results | Slots filled (of 10) |",
        "|---|---|---|---|",
    ]
    for s in SYSTEMS:
        e = summary[s]
        lines.append(f"| {s} | {e['ndcg@10']['mean']:.3f} [{e['ndcg@10']['low']:.3f}, {e['ndcg@10']['high']:.3f}] | "
                     f"{e['violation_rate']['mean']:.1%} | {e['fill_rate']['mean'] * K:.1f} |")
    lines.append("")
    for s in ("no_parsing", "post_filter"):
        d = summary["pre_filter"][f"ndcg_vs_{s}"]
        lines.append(f"- pre_filter vs {s}: nDCG@10 {d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]")
    markdown = "\n".join(lines)
    print("\n" + markdown + "\n")

    REPORTS_DIR.mkdir(exist_ok=True)
    stem = f"filters_{args.split}_{_git_sha()}_{time.strftime('%Y%m%d-%H%M%S')}"
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps({"summary": summary, "per_query": rows}, indent=2), encoding="utf-8")
    (REPORTS_DIR / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    logger.info(f"Reports written to reports/{stem}.json and .md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--collection", default=settings.COLLECTION_NAME)
    asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
