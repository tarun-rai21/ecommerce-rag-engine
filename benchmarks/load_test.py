"""Load test for the running API: latency and throughput under concurrent users.

    python -m benchmarks.load_test                      # all three parts (about 15 minutes)
    python -m benchmarks.load_test --parts search       # /search only, no LLM calls

Three parts, because Groq's free tier allows only a few uncached /ask calls a minute:

1. search:   /search under 1-32 concurrent users. Everything the project runs itself:
             query parsing, embedding, Qdrant hybrid search, RRF.
2. ask-miss: a paced, one-at-a-time sample of uncached /ask requests. Measures what the
             LLM call adds and records its token usage (for cost). Flushes the Redis cache
             first so every request is a miss.
3. ask-hit:  /ask under 1-32 concurrent users, replaying the ask-miss questions, so every
             request is a cache hit.

Users are closed-loop: each sends a request, waits for the answer, then sends the next,
with no think time. Each concurrency level runs for a warm-up period (not recorded) and a
measured period. Queries are WANDS test-split queries, shuffled with a fixed seed.

The client runs on the same machine as the stack, so it competes for CPU; treat the
numbers as a lower bound on what the server alone could do.
"""
import argparse
import asyncio
import itertools
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from loguru import logger

from src.evaluation.datasets import load_wands_queries
from src.evaluation.retrieval_eval import _git_sha

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
SEED = 11
PARTS = ("search", "ask-miss", "ask-hit")


@dataclass
class Sample:
    start: float
    latency_ms: float
    ok: bool
    status: str           # HTTP status, or the exception name
    body: dict = field(default_factory=dict)


# ── Requests ──────────────────────────────────────────────
async def send(client: httpx.AsyncClient, path: str, payload: dict) -> Sample:
    start = time.perf_counter()
    try:
        r = await client.post(path, json=payload)
        ok = r.status_code == 200
        body = r.json() if ok else {}
        status = str(r.status_code)
        if body.get("source") == "llm_error":  # HTTP 200, but the shopper got the "try again later" fallback
            ok, status = False, "llm_error"
    except httpx.HTTPError as e:
        ok, body, status = False, {}, type(e).__name__
    return Sample(start, (time.perf_counter() - start) * 1000, ok, status, body)


async def closed_loop(base_url: str, path: str, payloads: list[dict], users: int,
                      warmup_s: float, duration_s: float) -> tuple[list[Sample], float]:
    """`users` concurrent request loops; returns the samples started in the measured window."""
    queue = itertools.cycle(payloads)
    samples: list[Sample] = []
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(base_url=base_url, timeout=60, limits=limits) as client:
        t0 = time.perf_counter()
        measure_from, stop_at = t0 + warmup_s, t0 + warmup_s + duration_s

        async def user():
            while time.perf_counter() < stop_at:
                s = await send(client, path, next(queue))
                if s.start >= measure_from:
                    samples.append(s)

        await asyncio.gather(*(user() for _ in range(users)))
    return samples, duration_s


# ── Statistics ────────────────────────────────────────────
def percentile_ci(x: np.ndarray, q: float, n: int = 2000) -> tuple[float, float, float]:
    """Percentile with a 95% bootstrap CI (requests resampled with replacement)."""
    rng = np.random.default_rng(SEED)
    boot = np.percentile(x[rng.integers(0, x.size, size=(n, x.size))], q, axis=1)
    return float(np.percentile(x, q)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_level(users: int, samples: list[Sample], duration_s: float, server_field) -> dict:
    ok = [s for s in samples if s.ok]
    lat = np.array([s.latency_ms for s in ok]) if ok else np.array([])
    row = {
        "users": users, "requests": len(samples), "errors": len(samples) - len(ok),
        "error_statuses": sorted({s.status for s in samples if not s.ok}),
        "rps": len(ok) / duration_s,
    }
    if lat.size:
        for q in (50, 95, 99):
            row[f"p{q}"] = percentile_ci(lat, q)
        server = [server_field(s.body) for s in ok if server_field(s.body) is not None]
        row["server_p50"] = float(np.median(server)) if server else None
    return row


# ── Parts ─────────────────────────────────────────────────
async def run_levels(args, path: str, payloads: list[dict], server_field, check=None) -> list[dict]:
    rows = []
    for users in args.users:
        samples, duration = await closed_loop(args.base_url, path, payloads, users, args.warmup, args.duration)
        if check:
            check(samples)
        row = summarize_level(users, samples, duration, server_field)
        rows.append(row)
        p95 = row.get("p95", (float("nan"),))[0]
        logger.info(f"{path} users={users}: {row['rps']:.1f} req/s, p95 {p95:.0f} ms, errors {row['errors']}")
    return rows


async def ask_miss(args, payloads: list[dict]) -> list[dict]:
    if args.flush_cache:
        import redis
        redis.Redis(host=args.redis_host, port=args.redis_port).flushdb()
        logger.info("Redis cache flushed: every /ask below is a miss")
    rows = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=120) as client:
        for i, payload in enumerate(payloads, start=1):
            s = await send(client, "/api/v1/ask", payload)
            rows.append({"q": payload["q"], "latency_ms": s.latency_ms, "ok": s.ok, "status": s.status,
                         "source": s.body.get("source"), "timing": s.body.get("timing"), "usage": s.body.get("usage")})
            logger.info(f"ask-miss {i}/{len(payloads)}: {s.body.get('source')} {s.latency_ms:.0f} ms")
            if i < len(payloads):
                await asyncio.sleep(args.pace)  # stay under Groq's free-tier tokens/minute
    return rows


def require_cache_hits(samples: list[Sample]) -> None:
    misses = sum(s.ok and s.body.get("source") != "cache" for s in samples)
    if misses:
        logger.warning(f"{misses} /ask requests were not cache hits; the cache TTL may have expired")


# ── Report ────────────────────────────────────────────────
def fmt_ci(t) -> str:
    return f"{t[0]:.0f} [{t[1]:.0f}, {t[2]:.0f}]"


def level_table(rows: list[dict], server_label: str) -> list[str]:
    lines = [f"| Users | Requests/s | p50 ms | p95 ms | p99 ms | {server_label} p50 ms | Errors |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "p50" not in r:
            lines.append(f"| {r['users']} | 0 | - | - | - | - | {r['errors']} of {r['requests']} |")
            continue
        server = "-" if r["server_p50"] is None else f"{r['server_p50']:.0f}"
        errors = f"{r['errors']} of {r['requests']}" + (f" ({', '.join(r['error_statuses'])})" if r["errors"] else "")
        lines.append(f"| {r['users']} | {r['rps']:.1f} | {fmt_ci(r['p50'])} | {fmt_ci(r['p95'])} | "
                     f"{fmt_ci(r['p99'])} | {server} | {errors} |")
    return lines


def miss_summary(rows: list[dict]) -> list[str]:
    llm = [r for r in rows if r["source"] == "llm"]
    other = {}
    for r in rows:
        if r["source"] != "llm":
            other[r["source"] or r["status"]] = other.get(r["source"] or r["status"], 0) + 1
    if not llm:
        return [f"No successful LLM answers ({other})."]
    def med(xs):
        return float(np.median(xs))

    def t(key):
        return [r["timing"][key] for r in llm]

    usage = [r["usage"] for r in llm if r["usage"]]
    lines = [f"{len(llm)} of {len(rows)} requests answered by the LLM" + (f"; others: {other}" if other else "") + ".", "",
             "| Stage | Median ms | Max ms |", "|---|---|---|"]
    for label, key in (("Cache lookup", "cache_lookup_ms"), ("Search", "search_ms"), ("LLM call", "llm_ms"), ("Total (server)", "total_ms")):
        lines.append(f"| {label} | {med(t(key)):.0f} | {max(t(key)):.0f} |")
    lines.append(f"| Total (client) | {med([r['latency_ms'] for r in llm]):.0f} | {max(r['latency_ms'] for r in llm):.0f} |")
    if usage:
        lines += ["", f"Tokens per LLM answer (n={len(usage)}): prompt median {med([u['prompt_tokens'] for u in usage]):.0f}, "
                      f"completion median {med([u['completion_tokens'] for u in usage]):.0f} "
                      f"(max {max(u['completion_tokens'] for u in usage)}, includes reasoning)."]
    return lines


def to_markdown(report: dict) -> str:
    cfg = report["config"]
    lines = [f"Load test, commit `{report['commit']}`. Closed-loop users, {cfg['warmup']:.0f} s warm-up + "
             f"{cfg['duration']:.0f} s measured per level; WANDS test queries. Latency in ms, "
             "p50/p95/p99 with 95% bootstrap CIs. Client and stack on the same laptop.", ""]
    if "search" in report:
        lines += ["### /search", ""] + level_table(report["search"], "Server") + [""]
    if "ask-miss" in report:
        lines += ["### /ask, cache miss (one at a time, paced for Groq)", ""] + miss_summary(report["ask-miss"]) + [""]
    if "ask-hit" in report:
        lines += ["### /ask, cache hit", ""] + level_table(report["ask-hit"], "Server") + [""]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────
async def main_async(args) -> dict:
    queries = [q.text for q in load_wands_queries(split="test")]
    random.Random(SEED).shuffle(queries)
    report = {"commit": _git_sha(), "config": {k: v for k, v in vars(args).items() if k != "parts"}, "parts": args.parts}

    if "search" in args.parts:
        payloads = [{"q": q, "limit": 10} for q in queries]
        report["search"] = await run_levels(args, "/api/v1/search", payloads, lambda b: b.get("latency_ms"))
    ask_payloads = [{"q": q, "limit": 5} for q in queries[:args.ask_queries]]
    if "ask-miss" in args.parts:
        report["ask-miss"] = await ask_miss(args, ask_payloads)
    if "ask-hit" in args.parts:
        report["ask-hit"] = await run_levels(args, "/api/v1/ask", ask_payloads,
                                             lambda b: (b.get("timing") or {}).get("total_ms"), require_cache_hits)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    p.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--ask-queries", type=int, default=20, help="questions for the ask-miss and ask-hit parts")
    p.add_argument("--pace", type=float, default=20.0, help="seconds between ask-miss requests")
    p.add_argument("--flush-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="flush Redis before ask-miss so every request is a miss")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    args = p.parse_args(argv)

    report = asyncio.run(main_async(args))
    markdown = to_markdown(report)
    print("\n" + markdown)
    REPORTS_DIR.mkdir(exist_ok=True)
    stem = f"load_{report['commit']}_{time.strftime('%Y%m%d-%H%M%S')}"
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS_DIR / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    logger.info(f"Reports written to reports/{stem}.json and .md")


if __name__ == "__main__":
    main()
