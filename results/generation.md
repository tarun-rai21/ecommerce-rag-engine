Items: 110 (50 plain, 20 constrained, 30 ooc, 10 sized; plain and constrained are WANDS dev queries, constrained with a budget appended). Generator `openai/gpt-oss-20b`, judge `qwen/qwen3.8-27b`.

| Metric | Value [95% CI] |
|---|---|
| Correct refusal, out-of-catalog | 96.7% [90.0%, 100.0%] (n=30) |
| False refusal, in-catalog | 18.6% [10.0%, 28.6%] (n=70) |
| False refusal when an Exact product was retrieved | 1.9% [0.0%, 5.7%] (n=53) |
| Faithfulness (supported claims share) | 0.991 [0.985, 0.997] (n=68) |
| Fully faithful answers (every claim supported) | 86.8% [77.9%, 94.1%] (n=68) |
| Answer relevance (1-5, in-catalog) | 4.851 [4.761, 4.925] (n=67) |
| Answers quoting a price not in the context | 0.0% [0.0%, 0.0%] (n=68) |
| Answers with a whole-catalog claim ("the only ones") removed in code | 22.1% [13.2%, 32.4%] (n=68) |
| Answers still claiming to cover the whole catalog (pattern check) | 0.0% [0.0%, 0.0%] (n=68) |
| Answers mentioning "the context" to the shopper | 0.0% [0.0%, 0.0%] (n=68) |

- Claims checked: 1622, unsupported: 13. Judge parse errors: 0.
- Refusal rate by segment: plain 16%, constrained 25%, ooc 97%, sized 0%
- Relevance distribution (in-catalog answers): {"1": 0, "2": 0, "3": 0, "4": 10, "5": 57}
