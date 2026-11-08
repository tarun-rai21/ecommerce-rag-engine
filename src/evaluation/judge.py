"""LLM-as-judge and deterministic checks for generated answers.

Judge: qwen/qwen3.8-27b, a different model family from the gpt-oss-20b generator,
so the judge doesn't share the generator's blind spots. One judge call per answer
returns the answer's atomic claims (each marked supported / unsupported by the
retrieved context) and a 1-5 relevance score.

Every LLM call (generator and judge) goes through `CachedLLM`: responses are
stored in SQLite keyed by the full request, so re-running the eval is free and
reproducible, and rate-limit errors are waited out using Groq's retry-after.
"""
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from groq import APIConnectionError, Groq, InternalServerError, RateLimitError
from loguru import logger

from src.backend.services.llm import CATALOG_CLAIM, REFUSAL_MESSAGE

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "llm_calls.sqlite"
JUDGE_MODEL = "qwen/qwen3.8-27b"

JUDGE_SYSTEM = """You audit answers from a shopping assistant. The assistant may only use the PRODUCT CONTEXT it was given.

Do two things.

1. CLAIMS. Split the ANSWER into atomic factual claims about products: names, prices, ratings, review counts, materials, colors, sizes, styles, categories, features, and statements that a product meets the shopper's request or filters. Ignore greetings, offers to help, and generic phrasing ("here are some options"). For each claim decide whether the PRODUCT CONTEXT supports it.
   - supported = true only if the context states it or it follows directly from the context (e.g. "$89.99 is under $100").
   - If the context opens with a line saying all products satisfy the shopper's filters, claims that products meet those filters are supported.
   - Claims that go beyond the context (invented features, dimensions, stock, shipping, comparisons not in the data) are unsupported, even if plausible.

2. RELEVANCE. Score 1-5 how well the ANSWER addresses the QUESTION:
   5 = directly answers with suitable products; 4 = answers with minor gaps or one off-target item;
   3 = partially addresses it; 2 = mostly off-target; 1 = does not address it.

If the ANSWER is only a refusal ("I cannot find a matching product..."), return an empty claims list and relevance null.

Respond with JSON only:
{"claims": [{"claim": "...", "supported": true}], "relevance": 1-5 or null}"""


def judge_messages(query: str, context: str, answer: str) -> list[dict]:
    user = f"PRODUCT CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n{answer}"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


# Fallback: on very long answers (25+ claims) the judge sometimes omits the
# relevance score, so it is asked for separately, without the claim audit.
RELEVANCE_SYSTEM = """Score 1-5 how well the ANSWER from a shopping assistant addresses the shopper's QUESTION:
5 = directly answers with suitable products; 4 = answers with minor gaps or one off-target item;
3 = partially addresses it; 2 = mostly off-target; 1 = does not address it.
Respond with JSON only: {"relevance": 1-5}"""


def relevance_messages(query: str, answer: str) -> list[dict]:
    return [{"role": "system", "content": RELEVANCE_SYSTEM},
            {"role": "user", "content": f"QUESTION:\n{query}\n\nANSWER:\n{answer}"}]


@dataclass
class Judgment:
    claims: list[dict] = field(default_factory=list)  # [{"claim": str, "supported": bool}]
    relevance: int | None = None
    parse_error: str | None = None

    @property
    def faithfulness(self) -> float | None:
        """Supported claims / all claims; None when there is nothing to check."""
        if not self.claims:
            return None
        return sum(bool(c["supported"]) for c in self.claims) / len(self.claims)

    @property
    def fully_faithful(self) -> bool | None:
        return None if not self.claims else all(c["supported"] for c in self.claims)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgment(text: str) -> Judgment:
    """Tolerant parse: strips code fences / think blocks, validates field types."""
    m = _JSON_OBJECT.search(re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL))
    if not m:
        return Judgment(parse_error="no JSON object in judge output")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Judgment(parse_error=f"invalid JSON: {e}")
    claims = []
    for c in data.get("claims") or []:
        if isinstance(c, dict) and isinstance(c.get("supported"), bool):
            claims.append({"claim": str(c.get("claim", "")), "supported": c["supported"]})
    rel = data.get("relevance")
    if isinstance(rel, str) and rel.strip().isdigit():
        rel = int(rel)
    relevance = rel if isinstance(rel, int) and 1 <= rel <= 5 else None
    return Judgment(claims=claims, relevance=relevance)


# ── Deterministic checks ──────────────────────────────────
_REFUSAL_CORE = "cannot find a matching product"


def is_refusal(answer: str) -> bool:
    """True when the answer is (or contains) the system's refusal sentence."""
    norm = re.sub(r"\s+", " ", answer.lower().replace("’", "'"))
    return _REFUSAL_CORE in norm or REFUSAL_MESSAGE.lower() in norm


def overclaims_catalog(answer: str) -> bool:
    """Claims about the whole catalog ("these are the only ones…") that 5 retrieved
    products can't support. Same pattern production uses to remove them, so on
    final answers this counts only claims inside table rows."""
    return bool(CATALOG_CLAIM.search(answer))


_CONTEXT_LEAK = re.compile(r"\b(?:the|provided|given|supplied) (?:product )?(?:context|data)\b", re.IGNORECASE)


def mentions_context(answer: str) -> bool:
    """Answer exposes the pipeline to the shopper ("in the provided context")."""
    return bool(_CONTEXT_LEAK.search(answer))


_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")


def dollar_amounts(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "") + "." + (m.group(2) or "0")) for m in _MONEY.finditer(text)]


def ungrounded_prices(answer: str, context: str) -> list[float]:
    """Dollar amounts in the answer that appear nowhere in the context.

    Answers often round ("about $90"), so an amount also counts as grounded when
    it is the rounded (to whole dollars) form of a context price."""
    known = dollar_amounts(context)
    grounded = set(round(p, 2) for p in known) | {float(round(p)) for p in known} | {float(int(p)) for p in known}
    return [a for a in dollar_amounts(answer) if round(a, 2) not in grounded]


# ── Cached, rate-limit-aware LLM calls ────────────────────
class CachedLLM:
    def __init__(self, client: Groq, cache_path: Path = DEFAULT_CACHE, max_wait_s: float = 90.0, max_tries: int = 30):
        self.client, self.max_wait_s, self.max_tries = client, max_wait_s, max_tries
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self.db.commit()
        self.hits = self.misses = 0

    @staticmethod
    def key(model: str, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, model: str, messages: list[dict], **params) -> str:
        k = self.key(model, messages, params)
        row = self.db.execute("SELECT response FROM calls WHERE key = ?", (k,)).fetchone()
        if row:
            self.hits += 1
            return row[0]
        self.misses += 1
        for attempt in range(1, self.max_tries + 1):
            try:
                resp = self.client.chat.completions.create(model=model, messages=messages, **params)
                text = resp.choices[0].message.content or ""
                break
            except RateLimitError as e:
                wait = _retry_after(e, default=15.0 * attempt)
                logger.info(f"rate limited by {model}; waiting {wait:.0f}s (attempt {attempt})")
                time.sleep(min(wait, self.max_wait_s))
            except (APIConnectionError, InternalServerError) as e:
                # 5xx "over capacity" and dropped connections: exponential backoff.
                wait = min(10.0 * 2 ** (attempt - 1), 300.0)
                logger.warning(f"{model} unavailable ({type(e).__name__}); retrying in {wait:.0f}s (attempt {attempt})")
                time.sleep(wait)
        else:
            raise RuntimeError(f"{model}: gave up after {self.max_tries} attempts")
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (k, text))
        self.db.commit()
        return text

    def close(self) -> None:
        self.db.close()


def _retry_after(err: RateLimitError, default: float) -> float:
    try:
        return float(err.response.headers.get("retry-after")) + 1.0
    except Exception:
        return default
