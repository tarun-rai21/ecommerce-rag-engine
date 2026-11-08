"""Offline evaluation against a running backend.

Retrieval metrics come from POST /api/v1/search (no cache, no LLM), so they are
deterministic and independent of cache state. Answer metrics come from
POST /api/v1/ask and are reported separately.

Usage (from repo root):
    python -m src.evaluation.eval                 # retrieval + answers
    python -m src.evaluation.eval --skip-answers   # retrieval only (no LLM calls)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

from src.evaluation.metrics import (
    hit_rate_at_k,
    keyword_coverage,
    max_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_PATH = REPO_ROOT / "src" / "evaluation" / "ground_truth.json"
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_BACKEND_URL = "http://localhost:8000"
TOP_K = 5
# CI gate. Hit rate is saturated on the 39-product demo set; Phase 1 moves the
# gate to nDCG@10 on held-out human-labelled queries.
HIT_RATE_THRESHOLD = 0.70


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _post(client: httpx.Client, path: str, query: str, limit: int) -> dict:
    response = client.post(path, json={"q": query, "limit": limit})
    response.raise_for_status()
    return response.json()


def evaluate_retrieval(client: httpx.Client, cases: list[dict], k: int) -> list[dict]:
    rows = []
    for idx, case in enumerate(cases, start=1):
        query = case["query"]
        relevant = {str(e) for e in case["expected_product_ids"]}
        logger.info(f"[retrieval {idx}/{len(cases)}] '{query}'")
        try:
            data = _post(client, "/api/v1/search", query, k)
            retrieved = [str(r["id"]) for r in data["results"]]
            error = None
        except Exception as e:  # count failures as zero instead of aborting the run
            logger.error(f"Query failed: '{query}' — {e!r}")
            retrieved, error = [], repr(e)
        rows.append({
            "query": query,
            "retrieved_ids": retrieved,
            "expected_ids": sorted(relevant),
            "hit_rate": hit_rate_at_k(retrieved, relevant, k),
            "precision": precision_at_k(retrieved, relevant, k),
            "max_precision": max_precision_at_k(relevant, k),
            "recall": recall_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant, k),
            "ndcg": ndcg_at_k(retrieved, {r: 1.0 for r in relevant}, k),
            "error": error,
        })
    return rows


def evaluate_answers(client: httpx.Client, cases: list[dict], k: int) -> list[dict]:
    rows = []
    for idx, case in enumerate(cases, start=1):
        query = case["query"]
        logger.info(f"[answer {idx}/{len(cases)}] '{query}'")
        try:
            data = _post(client, "/api/v1/ask", query, k)
            answer, source = data.get("answer", ""), data.get("source", "unknown")
        except Exception as e:
            logger.error(f"Query failed: '{query}' — {e!r}")
            answer, source = "", "error"
        rows.append({
            "query": query,
            "source": source,
            "keyword_coverage": keyword_coverage(answer, case["ideal_response_keywords"]),
            "answer": answer,
        })
    return rows


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def print_report(retrieval: list[dict], answers: list[dict], k: int) -> dict:
    print("\n" + "=" * 104)
    print(f"{'QUERY':<40} {f'HIT@{k}':>8} {f'P@{k}':>7} {f'R@{k}':>7} {'MRR':>7} {f'nDCG@{k}':>8}  RETRIEVED")
    print("=" * 104)
    for r in retrieval:
        q = r["query"][:38] + ".." if len(r["query"]) > 38 else r["query"]
        print(f"{q:<40} {r['hit_rate']:>8.2f} {r['precision']:>7.2f} {r['recall']:>7.2f} "
              f"{r['mrr']:>7.2f} {r['ndcg']:>8.2f}  {','.join(r['retrieved_ids'])}")

    summary = {
        "queries": len(retrieval),
        f"hit_rate@{k}": _mean(retrieval, "hit_rate"),
        f"precision@{k}": _mean(retrieval, "precision"),
        f"precision@{k}_ceiling": _mean(retrieval, "max_precision"),
        f"recall@{k}": _mean(retrieval, "recall"),
        f"mrr@{k}": _mean(retrieval, "mrr"),
        f"ndcg@{k}": _mean(retrieval, "ndcg"),
        "retrieval_errors": sum(1 for r in retrieval if r["error"]),
    }
    if answers:
        summary["keyword_coverage"] = _mean(answers, "keyword_coverage")
        summary["answer_sources"] = {
            s: sum(1 for a in answers if a["source"] == s) for s in sorted({a["source"] for a in answers})
        }

    print("-" * 104)
    for key, value in summary.items():
        print(f"  {key:<24}: {value:.4f}" if isinstance(value, float) else f"  {key:<24}: {value}")
    print("-" * 104)
    return summary


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--skip-answers", action="store_true", help="Skip /ask (no LLM calls).")
    args = parser.parse_args(argv)

    cases = load_ground_truth()
    logger.info(f"Evaluating {len(cases)} queries against {args.backend_url} (k={args.k}).")
    with httpx.Client(base_url=args.backend_url, timeout=30) as client:
        retrieval = evaluate_retrieval(client, cases, args.k)
        answers = [] if args.skip_answers else evaluate_answers(client, cases, args.k)

    summary = print_report(retrieval, answers, args.k)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"eval_{_git_sha()}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(
        {"summary": summary, "retrieval": retrieval, "answers": answers}, indent=2
    ), encoding="utf-8")
    logger.info(f"Report written to {report_path}")

    hit_rate = summary[f"hit_rate@{args.k}"]
    if summary["retrieval_errors"] or hit_rate < HIT_RATE_THRESHOLD:
        logger.error(
            f"Gate FAILED: hit_rate@{args.k}={hit_rate:.4f} (threshold {HIT_RATE_THRESHOLD}), "
            f"errors={summary['retrieval_errors']}."
        )
        sys.exit(1)
    logger.info(f"Gate passed: hit_rate@{args.k}={hit_rate:.4f} ≥ {HIT_RATE_THRESHOLD}.")


if __name__ == "__main__":
    main()
