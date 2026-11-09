"""Export shopper feedback from Redis to a JSON Lines file, with a summary.

    python deploy/export_feedback.py                 # local Redis (localhost:6379)
    python deploy/export_feedback.py --ask-url       # prompts for a managed Redis URL (e.g. Upstash)

Each line holds one rating and the interaction it refers to: query, interpretation,
retrieved product ids, the answer and its source. The Redis URL is read with a hidden
prompt and never printed, because it contains the password.
"""
import argparse
import getpass
import json
import time
from collections import Counter
from pathlib import Path

import redis

FEEDBACK_LIST = "rag:feedback"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ask-url", action="store_true", help="prompt for a redis:// or rediss:// URL")
    args = p.parse_args()

    if args.ask_url:
        client = redis.from_url(getpass.getpass("Redis URL (input hidden): ").strip(), decode_responses=True)
    else:
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    entries = [json.loads(x) for x in client.lrange(FEEDBACK_LIST, 0, -1)]

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"feedback_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")

    ratings = Counter(e["rating"] for e in entries)
    by_source = Counter((e["interaction"]["source"], e["rating"]) for e in entries)
    print(f"{len(entries)} ratings -> {out}")
    print(f"up {ratings['up']}, down {ratings['down']}"
          + (f" ({ratings['up'] / len(entries):.0%} positive)" if entries else ""))
    for (source, rating), n in sorted(by_source.items()):
        print(f"  {source:<10} {rating:<5} {n}")


if __name__ == "__main__":
    main()
