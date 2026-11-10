Load test, commit `4830e27`. Closed-loop users, 5 s warm-up + 30 s measured per level; WANDS test queries. Latency in ms, p50/p95/p99 with 95% bootstrap CIs. Client and stack on the same laptop.

### /ask, cache hit

| Users | Requests/s | p50 ms | p95 ms | p99 ms | Server p50 ms | Errors |
|---|---|---|---|---|---|---|
| 1 | 18.7 | 52 [52, 52] | 64 [62, 65] | 69 [67, 72] | 2 | 0 of 561 |
| 2 | 39.8 | 50 [50, 50] | 55 [55, 56] | 59 [58, 61] | 1 | 0 of 1193 |
| 4 | 75.1 | 51 [50, 51] | 65 [63, 67] | 102 [90, 128] | 1 | 0 of 2252 |
| 8 | 124.4 | 59 [59, 60] | 98 [95, 100] | 154 [140, 170] | 3 | 0 of 3733 |
| 16 | 122.5 | 116 [115, 117] | 207 [202, 212] | 409 [364, 417] | 22 | 0 of 3675 |
| 32 | 79.1 | 134 [131, 138] | 1545 [1452, 1667] | 2590 [2387, 2957] | 2 | 0 of 2373 |

