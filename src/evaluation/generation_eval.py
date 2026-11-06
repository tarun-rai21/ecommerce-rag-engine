"""Answer-quality evaluation: faithfulness, relevance, refusal accuracy.

Fixed sample (seeded, from the dev split so the test split stays untouched):
  plain        50 WANDS dev queries with >=1 Exact product, as written
  constrained  20 other dev queries with a budget appended ("<query> under $X",
               X = median price of the query's Exact products, as in filter_eval)
  ooc          30 hand-written out-of-catalog queries (out_of_catalog.json):
               the catalog stocks nothing that matches, so the right answer is the
               refusal sentence

Each query runs through the production path: query parsing -> hybrid search
(limit 5) -> build_context -> generator (same prompt and parameters as /ask).
Then:
  - refusal       deterministic: does the answer contain the refusal sentence
  - prices        deterministic: every $ amount in the answer appears in the context
  - faithfulness  judge (qwen3.8-27b): supported claims / all claims
  - relevance     judge: 1-5

Judge calibration: `--export-labels` writes a blind sheet (no judge output) of 30
answered items for a reviewer to label; `--calibrate` reports judge-vs-reviewer agreement.

All LLM calls are cached in .cache/llm_calls.sqlite, so re-runs are free and
give identical numbers. A cold run takes ~30-40 min on Groq's free tier.

Usage:
  python -m src.evaluation.generation_eval
  python -m src.evaluation.generation_eval --check-ooc
  python -m src.evaluation.generation_eval --export-labels src/evaluation/judge_labels.csv
  python -m src.evaluation.generation_eval --calibrate src/evaluation/judge_labels.csv

--export-labels and --calibrate read a saved report (latest, or --report PATH) and
make no LLM calls.
"""
import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from pathlib import Path

from groq import Groq
from loguru import logger

from config.settings import settings
from src.backend.api.v1.endpoints import build_context, resolve_query
from src.backend.models.api_schemas import SearchQueryRequest
from src.backend.services.llm import GENERATION_PARAMS, REFUSAL_MESSAGE, build_messages, clean_output
from src.backend.services.search import ProductSearchService
from src.evaluation.datasets import load_wands_queries
from src.evaluation.filter_eval import budget_for, fetch_prices
from src.evaluation.judge import (
    JUDGE_MODEL, CachedLLM, is_refusal, judge_messages, mentions_context, overclaims_catalog, parse_judgment, relevance_messages,
    ungrounded_prices,
)
from src.evaluation.retrieval_eval import MemoEmbedder, _git_sha
from src.evaluation.stats import bootstrap_mean
from src.ingestion.embedder import ProductEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
OOC_PATH = Path(__file__).with_name("out_of_catalog.json")
SAMPLE_SEED = 7
N_PLAIN, N_CONSTRAINED = 50, 20
TOP_K = 5
JUDGE_PARAMS = {"temperature": 0.0, "max_tokens": 1200, "response_format": {"type": "json_object"}}


# ── Sample ────────────────────────────────────────────────
async def build_sample(service: ProductSearchService) -> list[dict]:
    dev = [q for q in load_wands_queries(split="dev") if q.exact]
    random.Random(SAMPLE_SEED).shuffle(dev)
    plain, constrained = dev[:N_PLAIN], dev[N_PLAIN:N_PLAIN + N_CONSTRAINED]
    prices = await fetch_prices(service, {p for q in constrained for p in q.exact})

    items = [{"item_id": f"plain:{q.query_id}", "segment": "plain", "query": q.text, "exact": sorted(q.exact)}
             for q in plain]
    for q in constrained:
        budget = budget_for([prices[p] for p in q.exact if p in prices])
        items.append({"item_id": f"constrained:{q.query_id}", "segment": "constrained",
                      "query": f"{q.text} under ${budget:.0f}", "exact": sorted(q.exact)})
    ooc = json.loads(OOC_PATH.read_text(encoding="utf-8"))["queries"]
    items += [{"item_id": f"ooc:{i}", "segment": "ooc", "query": text, "exact": []} for i, text in enumerate(ooc)]
    return items


# ── Run ───────────────────────────────────────────────────
def generator_params() -> dict:
    extra = {"reasoning_effort": settings.LLM_REASONING_EFFORT} if settings.LLM_REASONING_EFFORT else {}
    return {**GENERATION_PARAMS, **extra}


async def evaluate_item(item: dict, service: ProductSearchService, llm: CachedLLM) -> dict:
    resolved = resolve_query(SearchQueryRequest(q=item["query"], limit=TOP_K))
    results = await service.search(resolved.text, limit=TOP_K, filters=resolved.filters)
    row = {**item, "retrieved": [str(r["id"]) for r in results],
           "retrieved_titles": [r.get("title", "") for r in results]}
    row["exact_in_context"] = bool(set(row["retrieved"]) & set(item["exact"]))

    if not results:  # the API answers with the refusal without calling the LLM
        row.update(context="", answer=REFUSAL_MESSAGE, refused=True, ungrounded_prices=[],
                   claims=[], relevance=None, faithfulness=None, fully_faithful=None, judge_error=None)
        return row

    context = build_context(results, resolved.filters)
    answer = clean_output(llm.complete(settings.LLM_MODEL, build_messages(item["query"], context), **generator_params()))
    row.update(context=context, answer=answer, refused=is_refusal(answer),
               ungrounded_prices=ungrounded_prices(answer, context))

    if row["refused"]:
        row.update(claims=[], relevance=None, faithfulness=None, fully_faithful=None, judge_error=None)
        return row
    judgment = parse_judgment(llm.complete(JUDGE_MODEL, judge_messages(item["query"], context, answer), **JUDGE_PARAMS))
    if judgment.relevance is None and judgment.parse_error is None:
        judgment.relevance = parse_judgment(
            llm.complete(JUDGE_MODEL, relevance_messages(item["query"], answer), **JUDGE_PARAMS)).relevance
    row.update(claims=judgment.claims, relevance=judgment.relevance, faithfulness=judgment.faithfulness,
               fully_faithful=judgment.fully_faithful, judge_error=judgment.parse_error)
    return row


async def run_all(collection: str) -> list[dict]:
    service = ProductSearchService(embedder=MemoEmbedder(ProductEmbedder()), collection_name=collection)
    llm = CachedLLM(Groq(api_key=settings.GROQ_API_KEY, max_retries=0, timeout=60))
    rows = []
    try:
        items = await build_sample(service)
        for i, item in enumerate(items, start=1):
            rows.append(await evaluate_item(item, service, llm))
            if i % 10 == 0:
                logger.info(f"{i}/{len(items)} items (cache hits {llm.hits}, new calls {llm.misses})")
    finally:
        await service.close()
        llm.close()
    return rows


# ── Summary ───────────────────────────────────────────────
def _rate(xs: list[bool]) -> dict:
    return {**vars(bootstrap_mean([float(x) for x in xs])), "n": len(xs)} if xs else {"mean": None, "n": 0}


def _mean(xs: list[float]) -> dict:
    return {**vars(bootstrap_mean(xs)), "n": len(xs)} if xs else {"mean": None, "n": 0}


def summarize(rows: list[dict]) -> dict:
    def seg(*names):
        return [r for r in rows if r["segment"] in names]

    in_cat, ooc = seg("plain", "constrained"), seg("ooc")
    answered = [r for r in rows if not r["refused"]]
    judged = [r for r in answered if r["faithfulness"] is not None]
    summary = {
        "n_items": len(rows),
        "refusal": {
            "ooc_correct_refusal": _rate([r["refused"] for r in ooc]),
            "in_catalog_false_refusal": _rate([r["refused"] for r in in_cat]),
            "false_refusal_when_exact_retrieved": _rate([r["refused"] for r in in_cat if r["exact_in_context"]]),
            "by_segment": {s: _rate([r["refused"] for r in seg(s)]) for s in ("plain", "constrained", "ooc")},
        },
        "faithfulness": {
            "mean_supported_share": _mean([r["faithfulness"] for r in judged]),
            "fully_faithful_rate": _rate([r["fully_faithful"] for r in judged]),
            "claims_total": sum(len(r["claims"]) for r in judged),
            "claims_unsupported": sum(not c["supported"] for r in judged for c in r["claims"]),
            "by_segment": {s: _mean([r["faithfulness"] for r in judged if r["segment"] == s])
                           for s in ("plain", "constrained", "ooc")},
        },
        "relevance": {
            "mean": _mean([float(r["relevance"]) for r in answered if r["relevance"] is not None and r["segment"] != "ooc"]),
            "distribution": {str(k): sum(r["relevance"] == k for r in answered if r["segment"] != "ooc") for k in range(1, 6)},
        },
        "prices": {
            "answers_with_ungrounded_price": _rate([bool(r["ungrounded_prices"]) for r in answered]),
        },
        "catalog_overclaim": _rate([overclaims_catalog(r["answer"]) for r in answered]),
        "mentions_context": _rate([mentions_context(r["answer"]) for r in answered]),
        "judge_parse_errors": sum(r["judge_error"] is not None for r in rows),
    }
    return summary


def to_markdown(s: dict) -> str:
    def pct(e):
        return "n/a" if e["mean"] is None else f"{e['mean']:.1%} [{e['low']:.1%}, {e['high']:.1%}] (n={e['n']})"

    def num(e):
        return "n/a" if e["mean"] is None else f"{e['mean']:.3f} [{e['low']:.3f}, {e['high']:.3f}] (n={e['n']})"

    r, f = s["refusal"], s["faithfulness"]
    lines = [
        f"Items: {s['n_items']} (50 plain + 20 budget-constrained WANDS dev queries, 30 out-of-catalog). "
        f"Generator `{settings.LLM_MODEL}`, judge `{JUDGE_MODEL}`.",
        "",
        "| Metric | Value [95% CI] |",
        "|---|---|",
        f"| Correct refusal, out-of-catalog | {pct(r['ooc_correct_refusal'])} |",
        f"| False refusal, in-catalog | {pct(r['in_catalog_false_refusal'])} |",
        f"| False refusal when an Exact product was retrieved | {pct(r['false_refusal_when_exact_retrieved'])} |",
        f"| Faithfulness (supported claims share) | {num(f['mean_supported_share'])} |",
        f"| Fully faithful answers (every claim supported) | {pct(f['fully_faithful_rate'])} |",
        f"| Answer relevance (1-5, in-catalog) | {num(s['relevance']['mean'])} |",
        f"| Answers quoting a price not in the context | {pct(s['prices']['answers_with_ungrounded_price'])} |",
        f"| Answers claiming to cover the whole catalog (\"the only ones\") | {pct(s['catalog_overclaim'])} |",
        f"| Answers mentioning \"the context\" to the shopper | {pct(s['mentions_context'])} |",
        "",
        f"- Claims checked: {f['claims_total']}, unsupported: {f['claims_unsupported']}. "
        f"Judge parse errors: {s['judge_parse_errors']}.",
        "- Refusal rate by segment: " + ", ".join(
            f"{k} {v['mean']:.0%}" for k, v in r["by_segment"].items() if v["mean"] is not None),
        "- Relevance distribution (in-catalog answers): " + json.dumps(s["relevance"]["distribution"]),
    ]
    return "\n".join(lines)


# ── Calibration ───────────────────────────────────────────
LABEL_FIELDS = ["item_id", "segment", "query", "context", "answer", "reviewer_fully_faithful", "reviewer_relevance", "notes", "reviewer"]


def load_report(path: Path | None = None) -> list[dict]:
    if path is None:
        reports = sorted(REPORTS_DIR.glob("generation_*.json"), key=lambda p: p.stat().st_mtime)
        if not reports:
            raise SystemExit("No generation report found: run the evaluation first.")
        path = reports[-1]
    logger.info(f"Using report {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def export_labels(rows: list[dict], path: Path, n: int = 30) -> None:
    """Blind sheet: the judge's verdicts are deliberately left out."""
    candidates = [r for r in rows if not r["refused"]]
    random.Random(SAMPLE_SEED).shuffle(candidates)
    ooc = [r for r in candidates if r["segment"] == "ooc"][:5]
    rest = [r for r in candidates if r["segment"] != "ooc"][: n - len(ooc)]
    chosen = sorted(ooc + rest, key=lambda r: r["item_id"])
    with path.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM: Excel then reads it as UTF-8
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in chosen:
            w.writerow({k: r.get(k, "") for k in LABEL_FIELDS[:5]})
    logger.info(f"Wrote {len(chosen)} items to {path}. Fill reviewer_fully_faithful (1/0) and reviewer_relevance (1-5).")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return float("nan")
    return statistics.correlation(ra, rb)


def calibrate(rows: list[dict], path: Path) -> str:
    by_id = {r["item_id"]: r for r in rows}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        labels = [l for l in csv.DictReader(fh) if l["reviewer_fully_faithful"].strip() and l["reviewer_relevance"].strip()]
    pairs = [(l, by_id[l["item_id"]]) for l in labels if l["item_id"] in by_id and by_id[l["item_id"]]["fully_faithful"] is not None]
    if not pairs:
        return "No labelled items to compare."
    hf = [int(l["reviewer_fully_faithful"]) for l, _ in pairs]
    jf = [int(r["fully_faithful"]) for _, r in pairs]
    rel = [(int(l["reviewer_relevance"]), r["relevance"]) for l, r in pairs if r["relevance"] is not None]
    lines = [
        f"Judge calibration on {len(pairs)} independently labelled answers:",
        f"- Fully faithful (yes/no): agreement {sum(x == y for x, y in zip(hf, jf)) / len(pairs):.0%}, "
        f"Cohen's kappa {cohen_kappa(hf, jf):.2f}; reviewer 'yes' rate {sum(hf) / len(hf):.0%}, judge {sum(jf) / len(jf):.0%}",
        f"- Relevance (1-5): exact agreement {sum(h == j for h, j in rel) / len(rel):.0%}, "
        f"within 1 point {sum(abs(h - j) <= 1 for h, j in rel) / len(rel):.0%}, "
        f"Spearman {spearman([h for h, _ in rel], [j for _, j in rel]):.2f}",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────
async def check_ooc(collection: str) -> None:
    service = ProductSearchService(embedder=ProductEmbedder(), collection_name=collection)
    try:
        for text in json.loads(OOC_PATH.read_text(encoding="utf-8"))["queries"]:
            res = await service.search(text, limit=3)
            print(f"{text!r}\n    " + "\n    ".join(f"{r.get('product_class')}: {r['title']}" for r in res))
    finally:
        await service.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", default=settings.COLLECTION_NAME)
    p.add_argument("--check-ooc", action="store_true", help="print top results for the out-of-catalog queries")
    p.add_argument("--export-labels", type=Path, help="write a blind labelling sheet to this CSV")
    p.add_argument("--calibrate", type=Path, help="compare judge verdicts with a filled-in label sheet")
    p.add_argument("--report", type=Path, help="report JSON for --export-labels / --calibrate (default: latest)")
    args = p.parse_args(argv)

    if args.check_ooc:
        asyncio.run(check_ooc(args.collection))
        return
    # Both work from a saved report, so labels always refer to the answers that were scored.
    if args.export_labels:
        export_labels(load_report(args.report), args.export_labels)
        return
    if args.calibrate:
        print("\n" + calibrate(load_report(args.report), args.calibrate) + "\n")
        return
    rows = asyncio.run(run_all(args.collection))
    summary = summarize(rows)
    markdown = to_markdown(summary)
    print("\n" + markdown + "\n")

    REPORTS_DIR.mkdir(exist_ok=True)
    stem = f"generation_{_git_sha()}_{time.strftime('%Y%m%d-%H%M%S')}"
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")
    (REPORTS_DIR / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    logger.info(f"Reports written to reports/{stem}.json and .md")


if __name__ == "__main__":
    main()
