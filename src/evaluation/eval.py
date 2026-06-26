import sys
import json
import httpx
from loguru import logger

BACKEND_URL = "http://localhost:8000"
GROUND_TRUTH_PATH = "src/evaluation/ground_truth.json"
TOP_K = 5
PRECISION_THRESHOLD = 0.70


# ── Load Ground Truth ─────────────────────────────────────
def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH, "r") as f:
        return json.load(f)


# ── Backend Query ─────────────────────────────────────────
def query_backend(query: str, limit: int = TOP_K) -> dict:
    response = httpx.post(
        f"{BACKEND_URL}/api/v1/ask",
        json={"q": query, "limit": limit},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ── Precision @ K ─────────────────────────────────────────
def compute_precision_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = TOP_K,
) -> float:
    retrieved_top_k = retrieved_ids[:k]
    relevant_count = sum(1 for rid in retrieved_top_k if rid in expected_ids)
    return relevant_count / k

# ── Hit Rate @ K ──────────────────────────────────────────
def compute_hit_rate_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int = TOP_K,
) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    return 1.0 if any(eid in retrieved_top_k for eid in expected_ids) else 0.0


# ── Faithfulness Check ────────────────────────────────────
def compute_faithfulness(
    answer: str,
    ideal_keywords: list[str],
) -> float:
    answer_lower = answer.lower()
    matched = sum(
        1 for keyword in ideal_keywords
        if keyword.lower() in answer_lower
    )
    return matched / len(ideal_keywords) if ideal_keywords else 0.0


# ── Terminal Table ────────────────────────────────────────
def print_results_table(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'QUERY':<40} {'PRECISION@5':>12} {'HIT RATE@5':>12} {'FAITHFULNESS':>14} {'SOURCE':>10}")
    print("=" * 100)
    for r in results:
        query_short = r["query"][:38] + ".." if len(r["query"]) > 38 else r["query"]
        print(
            f"{query_short:<40} "
            f"{r['precision_at_k']:>11.2f} "
            f"{r['hit_rate_at_k']:>11.2f} "
            f"{r['faithfulness']:>13.2f} "
            f"{r['source']:>10}"
        )
    print("=" * 100)


# ── Summary Report ────────────────────────────────────────
def print_summary(results: list[dict]) -> float:
    avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
    avg_hit_rate = sum(r["hit_rate_at_k"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    total_cache_hits = sum(1 for r in results if r["source"] == "cache")

    print(f"\n{'EVALUATION SUMMARY':^100}")
    print("-" * 100)
    print(f"  Total Queries Evaluated  : {len(results)}")
    print(f"  Average Precision @ {TOP_K}   : {avg_precision:.4f}")
    print(f"  Average Hit Rate @ {TOP_K}    : {avg_hit_rate:.4f}")
    print(f"  Average Faithfulness     : {avg_faithfulness:.4f}")
    print(f"  Cache Hits               : {total_cache_hits}/{len(results)}")
    print(f"  Hit Rate Threshold       : {PRECISION_THRESHOLD}")
    status = "✓ PASSED" if avg_hit_rate >= PRECISION_THRESHOLD else "✗ FAILED"
    print(f"  CI/CD Gate Status        : {status}")
    print("-" * 100)

    return avg_hit_rate


# ── Main Evaluation Loop ──────────────────────────────────
def main() -> None:
    logger.info("Starting evaluation suite.")
    ground_truth = load_ground_truth()
    results = []

    for idx, test_case in enumerate(ground_truth):
        query = test_case["query"]
        expected_ids = [str(eid) for eid in test_case["expected_product_ids"]]
        ideal_keywords = test_case["ideal_response_keywords"]

        logger.info(f"[{idx + 1}/{len(ground_truth)}] Evaluating: '{query}'")

        try:
            response_data = query_backend(query)

            retrieved_ids = [
                str(p["id"])
                for p in response_data.get("retrieved_products", [])
            ]

            answer = response_data.get("answer", "")
            source = response_data.get("source", "unknown")

            precision = compute_precision_at_k(retrieved_ids, expected_ids)
            hit_rate = compute_hit_rate_at_k(retrieved_ids, expected_ids)
            faithfulness = compute_faithfulness(answer, ideal_keywords)

            results.append({
                "query": query,
                "precision_at_k": precision,
                "hit_rate_at_k": hit_rate,
                "faithfulness": faithfulness,
                "source": source,
                "retrieved_ids": retrieved_ids,
                "expected_ids": expected_ids,
            })

        except Exception as e:
            logger.error(f"Query failed: '{query}' — {e}")
            results.append({
                "query": query,
                "precision_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "faithfulness": 0.0,
                "source": "error",
                "retrieved_ids": [],
                "expected_ids": expected_ids,
            })

    print_results_table(results)
    avg_precision = print_summary(results)

    if avg_precision < PRECISION_THRESHOLD:
        logger.error(f"Precision {avg_precision:.4f} below threshold {PRECISION_THRESHOLD}. CI/CD gate FAILED.")
        sys.exit(1)

    logger.info("Evaluation complete. All gates passed.")


if __name__ == "__main__":
    main()