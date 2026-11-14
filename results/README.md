# Results behind the numbers in the main README

Copies of the evaluation reports that the README quotes, so the numbers can be checked
without re-running anything. `reports/` (every run, gitignored) is where the scripts write;
a run is copied here only when the README is updated from it.

| File | Script | Run | README section |
|---|---|---|---|
| `retrieval_test.*` | `python -m src.evaluation.retrieval_eval --split test` | 2026-11-11 | Retrieval on WANDS |
| `filters_test.*` | `python -m src.evaluation.filter_eval --split test` | 2026-11-11 | Price-constrained retrieval |
| `size_test.md` | `python -m src.evaluation.size_eval` (prints only) | 2026-11-11 | Room-size rug queries |
| `generation.*` | `python -m src.evaluation.generation_eval` | 2026-11-14, commit `ef10e81` | Answer quality |
| `load_test.*` | `python -m benchmarks.load_test` | 2026-11-09 | Load test, cost per query |

The JSON files hold per-query rows: for retrieval, each query's top 10 ids and scores; for
generation, each question's retrieved context, answer and judge verdicts. The git hash in
the original file names was the checked-out commit; the 2026-11-11 runs were made with the
review fixes not yet committed.
