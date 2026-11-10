Split: **test** (301 queries). Mean with 95% bootstrap CI.

| Mode | nDCG@10 | Recall@100 (Exact) | MRR@10 (Exact) | P@10 (Exact) | judged@10 | p50 / p95 ms |
|---|---|---|---|---|---|---|
| dense | 0.672 [0.638, 0.706] | 0.622 [0.575, 0.665] | 0.652 [0.600, 0.702] | 0.451 [0.404, 0.500] | 0.79 | 15 / 30 |
| sparse | 0.659 [0.622, 0.696] | 0.659 [0.612, 0.701] | 0.723 [0.672, 0.771] | 0.460 [0.414, 0.506] | 0.79 | 9 / 16 |
| hybrid | 0.725 [0.692, 0.757] | 0.675 [0.629, 0.714] | 0.736 [0.686, 0.783] | 0.492 [0.444, 0.539] | 0.85 | 15 / 26 |

Paired difference vs **dense** (95% CI; excludes 0 → significant):

- sparse: ndcg@10 -0.013 [-0.042, +0.013] (not significant); recall@100 +0.037 [+0.006, +0.070] (significant)
- hybrid: ndcg@10 +0.053 [+0.036, +0.069] (significant); recall@100 +0.053 [+0.028, +0.077] (significant)

Exact-based metrics use the 237 queries that have at least one Exact judgment.
