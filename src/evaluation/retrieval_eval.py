"""Retrieval evaluation on WANDS with baselines and bootstrap confidence intervals.

Runs the production ProductSearchService in-process (no HTTP, no cache, no LLM)
for each retrieval mode and scores it against human relevance judgments.

Metrics (per query, then averaged over queries):
- nDCG@10   graded gains (Exact=2, Partial=1); PRIMARY metric
- Recall@100, MRR@10, P@10 with "relevant" = Exact; queries with no Exact
  judgment are excluded from these three (count reported)
- judged@10 share of the top 10 that has any human judgment. Unjudged results
  count as irrelevant, so a low judged@10 means nDCG may be underestimated.
- latency    retrieval only (Qdrant round trip, plus cross-encoder scoring for
  hybrid_rerank); query embedding is excluded because it is computed once per
  query and shared across modes.

Modes: dense, sparse, hybrid, hybrid_rerank (--rerank-model, --rerank-candidates).

Usage (from repo root, Qdrant running, collection ingested):
    python -m src.evaluation.retrieval_eval --split dev          # tune here
    python -m src.evaluation.retrieval_eval --split test         # report this
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.backend.services.rerank import RerankerService
from src.backend.services.search import ProductSearchService
from src.evaluation.datasets import EvalQuery, load_wands_queries
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.evaluation.stats import bootstrap_mean, paired_difference
from src.ingestion.embedder import ProductEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
K = 10
DEPTH = 100
METRICS = ("ndcg@10", "recall@100", "mrr@10", "p@10", "judged@10")
EXACT_ONLY = {"recall@100", "mrr@10", "p@10"}


class MemoEmbedder:
    """Embeds each query once even though several modes are evaluated."""

    def __init__(self, inner: ProductEmbedder) -> None:
        self.inner, self._memo = inner, {}

    def embed_query(self, text: str):
        if text not in self._memo:
            self._memo[text] = self.inner.embed_query(text)
        return self._memo[text]


def score_query(q: EvalQuery, top: list[str], deep: list[str]) -> dict[str, float | None]:
    """top: results of a limit=K search (production settings); deep: limit=DEPTH search.

    They are separate searches because hybrid fusion depends on how many candidates
    each leg fetches: asking for 100 results raises the per-leg prefetch, which can
    reorder the top 10 relative to what the API serves."""
    exact = q.exact
    return {
        "ndcg@10": ndcg_at_k(top, q.gains, K),
        "recall@100": recall_at_k(deep, exact, DEPTH) if exact else None,
        "mrr@10": reciprocal_rank(top, exact, K) if exact else None,
        "p@10": precision_at_k(top, exact, K) if exact else None,
        "judged@10": sum(1 for r in top[:K] if r in q.gains) / K,
    }


async def run_mode(service: ProductSearchService, queries: list[EvalQuery], mode: str) -> list[dict]:
    rows = []
    for i, q in enumerate(queries, start=1):
        service.embedder.embed_query(q.text)  # warm the memo so latency excludes embedding
        start = time.perf_counter()
        top = [str(r["id"]) for r in await service.search(q.text, limit=K, mode=mode)]
        latency_ms = (time.perf_counter() - start) * 1000
        # Reranking only reorders the top candidates (< DEPTH), so it can't change
        # which products are in the top 100: reuse the plain hybrid deep search.
        deep_mode = "hybrid" if mode == "hybrid_rerank" else mode
        deep = [str(r["id"]) for r in await service.search(q.text, limit=DEPTH, mode=deep_mode)]
        rows.append({"query_id": q.query_id, "query": q.text, "latency_ms": latency_ms,
                     "top10": top, **score_query(q, top, deep)})
        if i % 50 == 0:
            logger.info(f"[{mode}] {i}/{len(queries)} queries")
    return rows


def summarise(rows_by_mode: dict[str, list[dict]], baseline: str) -> dict:
    summary: dict = {}
    for mode, rows in rows_by_mode.items():
        entry = {}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            entry[metric] = vars(bootstrap_mean(values)) | {"n": len(values)}
        lat = sorted(r["latency_ms"] for r in rows)
        entry["latency_ms"] = {"p50": statistics.median(lat), "p95": lat[int(0.95 * (len(lat) - 1))]}
        if mode != baseline and baseline in rows_by_mode:
            base = {r["query_id"]: r for r in rows_by_mode[baseline]}
            entry["vs_baseline"] = {}
            for metric in ("ndcg@10", "recall@100"):
                pairs = [(r[metric], base[r["query_id"]][metric]) for r in rows
                         if r[metric] is not None and base[r["query_id"]][metric] is not None]
                entry["vs_baseline"][metric] = vars(paired_difference([a for a, _ in pairs], [b for _, b in pairs]))
        summary[mode] = entry
    return summary


def to_markdown(summary: dict, baseline: str, split: str, n_queries: int) -> str:
    lines = [
        f"Split: **{split}** ({n_queries} queries). Mean with 95% bootstrap CI.",
        "",
        "| Mode | nDCG@10 | Recall@100 (Exact) | MRR@10 (Exact) | P@10 (Exact) | judged@10 | p50 / p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, e in summary.items():
        cells = [f"{e[m]['mean']:.3f} [{e[m]['low']:.3f}, {e[m]['high']:.3f}]" for m in METRICS[:4]]
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | {e['judged@10']['mean']:.2f} | "
                     f"{e['latency_ms']['p50']:.0f} / {e['latency_ms']['p95']:.0f} |")
    diffs = [(m, e["vs_baseline"]) for m, e in summary.items() if "vs_baseline" in e]
    if diffs:
        lines += ["", f"Paired difference vs **{baseline}** (95% CI; excludes 0 → significant):", ""]
        for mode, vs in diffs:
            parts = []
            for metric, d in vs.items():
                sig = "significant" if d["low"] > 0 or d["high"] < 0 else "not significant"
                parts.append(f"{metric} {d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] ({sig})")
            lines.append(f"- {mode}: " + "; ".join(parts))
    exact_n = next(iter(summary.values()))["recall@100"]["n"]
    lines += ["", f"Exact-based metrics use the {exact_n} queries that have at least one Exact judgment."]
    return "\n".join(lines)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


async def main_async(args: argparse.Namespace) -> None:
    queries = load_wands_queries(split=args.split)
    if args.max_queries:
        queries = queries[: args.max_queries]
    logger.info(f"Evaluating {len(queries)} {args.split} queries on '{args.collection}' — modes: {args.modes}")

    reranker = None
    if "hybrid_rerank" in args.modes:
        reranker = RerankerService(args.rerank_model, candidates=args.rerank_candidates,
                                   timeout_seconds=args.rerank_timeout)
    service = ProductSearchService(
        embedder=MemoEmbedder(ProductEmbedder()),
        collection_name=args.collection,
        rrf_k=args.rrf_k,
        prefetch_limit=args.prefetch,
        reranker=reranker,
    )
    try:
        rows_by_mode = {mode: await run_mode(service, queries, mode) for mode in args.modes}
    finally:
        await service.close()

    summary = summarise(rows_by_mode, baseline=args.baseline)
    markdown = to_markdown(summary, args.baseline, args.split, len(queries))
    print("\n" + markdown + "\n")

    REPORTS_DIR.mkdir(exist_ok=True)
    stem = f"retrieval_{args.split}_{_git_sha()}_{time.strftime('%Y%m%d-%H%M%S')}"
    config = {"collection": args.collection, "split": args.split, "rrf_k": service.rrf_k,
              "prefetch": service.prefetch_limit, "native_fusion": service.use_native_fusion, "modes": args.modes,
              "rerank_model": args.rerank_model if reranker else None,
              "rerank_candidates": args.rerank_candidates if reranker else None}
    (REPORTS_DIR / f"{stem}.json").write_text(
        json.dumps({"config": config, "summary": summary, "per_query": rows_by_mode}, indent=2), encoding="utf-8")
    (REPORTS_DIR / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    logger.info(f"Reports written to reports/{stem}.json and .md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--modes", nargs="+", default=["dense", "sparse", "hybrid"])
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--collection", default=settings.COLLECTION_NAME)
    parser.add_argument("--rrf-k", type=int, default=settings.RRF_K)
    parser.add_argument("--prefetch", type=int, default=settings.PREFETCH_LIMIT,
                        help="Candidates per leg before fusion (raised to the eval depth automatically).")
    parser.add_argument("--max-queries", type=int, default=None, help="Quick runs: only the first N queries.")
    parser.add_argument("--rerank-model", default=settings.RERANK_MODEL)
    parser.add_argument("--rerank-candidates", type=int, default=settings.RERANK_CANDIDATES)
    parser.add_argument("--rerank-timeout", type=float, default=60.0,
                        help="Generous by default so evaluation measures quality, not timeouts.")
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
