"""Bootstrap confidence intervals over queries.

Queries are the unit of sampling: each resample draws queries with replacement
and recomputes the mean. Paired comparisons resample the *same* queries for both
systems, so per-query difficulty cancels out.
"""
from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 2000
SEED = 13


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float

    def fmt(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_mean(values: list[float], n: int = N_RESAMPLES, seed: int = SEED, alpha: float = 0.05) -> Estimate:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(x.mean()), float(lo), float(hi))


def paired_difference(a: list[float], b: list[float], **kw) -> Estimate:
    """CI of mean(a - b) over the same queries. Excludes 0 → difference is significant."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_mean([x - y for x, y in zip(a, b)], **kw)
