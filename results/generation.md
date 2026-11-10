Items: 110 (50 plain + 20 budget-constrained WANDS dev queries, 30 out-of-catalog, 10 rug queries with a size). Generator `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`.

| Metric | Value [95% CI] |
|---|---|
| Correct refusal, out-of-catalog | 100.0% [100.0%, 100.0%] (n=30) |
| False refusal, in-catalog | 14.3% [7.1%, 22.9%] (n=70) |
| False refusal when an Exact product was retrieved | 3.7% [0.0%, 9.3%] (n=54) |
| Faithfulness (supported claims share) | 0.990 [0.983, 0.996] (n=70) |
| Fully faithful answers (every claim supported) | 84.3% [75.7%, 92.9%] (n=70) |
| Answer relevance (1-5, in-catalog) | 4.786 [4.686, 4.871] (n=70) |
| Answers quoting a price not in the context | 0.0% [0.0%, 0.0%] (n=70) |
| Answers with a whole-catalog claim ("the only ones") removed in code | 14.3% [5.7%, 22.9%] (n=70) |
| Answers still claiming to cover the whole catalog (pattern check) | 0.0% [0.0%, 0.0%] (n=70) |
| Answers mentioning "the context" to the shopper | 0.0% [0.0%, 0.0%] (n=70) |

- Claims checked: 1701, unsupported: 14. Judge parse errors: 0.
- Refusal rate by segment: plain 12%, constrained 20%, ooc 100%, sized 0%
- Relevance distribution (in-catalog answers): {"1": 0, "2": 0, "3": 0, "4": 15, "5": 55}
