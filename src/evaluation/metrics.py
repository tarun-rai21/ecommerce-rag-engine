"""Ranking and answer metrics. Pure functions; ids are compared as strings."""
import math


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    # Denominator is k even if fewer results came back (missing slots count as misses).
    # Ceiling is min(|relevant|, k) / k — see max_precision_at_k.
    return sum(1 for r in retrieved[:k] if r in relevant) / k


def max_precision_at_k(relevant: set[str], k: int) -> float:
    return min(len(relevant), k) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    for i, r in enumerate(retrieved[:k] if k else retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gains: dict[str, float], k: int) -> float:
    """nDCG with exponential gain: DCG = Σ (2^g − 1) / log2(i + 1)."""
    dcg = sum(
        (2 ** gains.get(r, 0.0) - 1) / math.log2(i + 1)
        for i, r in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Share of expected keywords that appear in the answer (case-insensitive).

    NOT a faithfulness metric: it never looks at the retrieved context, so an
    answer can invent facts and still score 1.0. Grounding is evaluated separately
    with an LLM judge (roadmap 1.4)."""
    if not keywords:
        return 0.0
    lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)
