Split: **test** (237 queries with ≥1 Exact; budget = median price of the query's Exact products).

| System | nDCG@10 (budget-aware) | Over-budget results | Slots filled (of 10) |
|---|---|---|---|
| no_parsing | 0.364 [0.336, 0.392] | 50.2% | 10.0 |
| post_filter | 0.479 [0.447, 0.509] | 0.0% | 4.9 |
| post_filter_overfetch | 0.700 [0.662, 0.736] | 0.0% | 9.8 |
| pre_filter | 0.711 [0.674, 0.747] | 0.0% | 10.0 |

- pre_filter vs no_parsing: nDCG@10 +0.347 [+0.319, +0.375]
- pre_filter vs post_filter: nDCG@10 +0.232 [+0.210, +0.256]
- pre_filter vs post_filter_overfetch: nDCG@10 +0.011 [+0.003, +0.019]
