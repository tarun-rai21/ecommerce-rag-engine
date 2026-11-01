"""Labelled query sets for retrieval evaluation.

WANDS judgments: Exact (gain 2), Partial (gain 1), Irrelevant (gain 0).
Conventions (documented because they change the numbers):
- Only judged (query, product) pairs carry labels; unjudged products count as
  gain 0, the standard IR convention for pooled judgments.
- 1,467 (query, product) pairs are judged more than once; 14 of them disagree.
  We keep the maximum gain.
- Queries are split into dev (tuning) and test (reporting) by a hash of the
  query id, so the split is fixed and independent of file order.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from src.ingestion.loaders.wands import DEFAULT_DIR

GAINS = {"Exact": 2.0, "Partial": 1.0, "Irrelevant": 0.0}
SPLIT_SEED = "wands-split-v1"
DEV_FRACTION = 1 / 3
Split = Literal["dev", "test", "all"]


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    text: str
    query_class: str | None
    gains: dict[str, float] = field(default_factory=dict)  # product_id → gain (judged only)

    @property
    def exact(self) -> set[str]:
        return {pid for pid, g in self.gains.items() if g >= GAINS["Exact"]}


def split_of(query_id: str, seed: str = SPLIT_SEED, dev_fraction: float = DEV_FRACTION) -> str:
    h = int(hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_fraction else "test"


def load_wands_queries(data_dir: Path = DEFAULT_DIR, split: Split = "all") -> list[EvalQuery]:
    queries = pd.read_csv(data_dir / "query.csv", sep="\t", encoding="utf-8")
    labels = pd.read_csv(data_dir / "label.csv", sep="\t", encoding="utf-8")
    labels["gain"] = labels["label"].map(GAINS)
    if labels["gain"].isna().any():
        raise ValueError(f"Unknown labels: {set(labels.loc[labels.gain.isna(), 'label'])}")

    best = labels.groupby(["query_id", "product_id"], as_index=False)["gain"].max()
    gains_by_query: dict[int, dict[str, float]] = {}
    for qid, pid, gain in best.itertuples(index=False):
        gains_by_query.setdefault(int(qid), {})[str(pid)] = float(gain)

    result = []
    for row in queries.itertuples(index=False):
        qid = str(row.query_id)
        if split != "all" and split_of(qid) != split:
            continue
        query_class = row.query_class if isinstance(row.query_class, str) else None
        result.append(EvalQuery(qid, str(row.query).strip(), query_class, gains_by_query.get(int(qid), {})))
    return result
