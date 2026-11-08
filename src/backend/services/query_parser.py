"""Extract price and rating constraints from a free-text query.

"grey area rug under $300"  →  text="grey area rug", max_price=300
"sofa between $500 and $900" →  text="sofa", min_price=500, max_price=900
"desk lamp 4+ stars"         →  text="desk lamp", min_rating=4.0

Why rules, not an LLM: constraints are numeric patterns, rules are instant, free,
deterministic and testable, and they don't depend on a rate-limited API. The cost
of a false positive (filtering out good results) is high, so the rules are strict:

- A number is a price only with a currency marker ($, dollars, usd, bucks) or when
  it directly follows a price phrase (under, less than, between …) and is NOT
  followed by a unit (inch, ft, drawer, light, piece …). "48 inch vanity",
  "twin over full bunk beds" and "end table between recliners" yield nothing.
- A rating needs an explicit "N stars", "N+ stars" or "rated N+". "star wars rug"
  yields nothing.

Categories are deliberately NOT inferred from text: a wrong category filter hides
good results, and ranking already handles category semantics. Categories come
from explicit facet filters (see SearchFilters.categories).
"""
import re

from pydantic import BaseModel, Field, model_validator

# Lookarounds stop the engine matching part of a number: without them, backtracking
# turns "under 30 inches" into "under 3" (the next char '0' isn't a unit).
_NUM = r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?![\d.,]?\d)"
_CURRENCY_BEFORE = r"(?:\$|usd\s*)"
_CURRENCY_AFTER = r"(?:\s*(?:dollars?|usd|bucks))"
# Units that make a number a size/count rather than a price.
_UNIT_AFTER = (
    r"(?!\s*(?:\"|'|in\b|inch|inches|ft\b|feet|foot|cm\b|mm\b|m\b|qt\b|quart|lbs?\b|pounds?|oz\b|"
    r"gallons?|piece|pc\b|pcs\b|drawers?|draw\b|lights?|seats?|seater|people|person|x\s*\d|-?\s*\d+/\d+|"
    r"stars?\b|years?\b|yr\b|v\b|volt|watt|w\b|gang\b|tier|shelf|shelves|door|panel))"
)

# A price: "$200", "200 dollars", or a bare number only in price-phrase context.
_PRICE_STRICT = rf"(?:{_CURRENCY_BEFORE}\s*{_NUM}|{_NUM}{_CURRENCY_AFTER})"
_PRICE_LOOSE = rf"(?:{_CURRENCY_BEFORE}?\s*{_NUM}{_CURRENCY_AFTER}?){_UNIT_AFTER}"

_MAX_WORDS = r"(?:under|below|less than|cheaper than|up to|no more than|at most|max(?:imum)?|within)"
_MIN_WORDS = r"(?:over|above|more than|at least|min(?:imum)?|starting at|from)"
_FILLER = r"(?:\s*(?:priced|costing|for|that costs?|price|budget|of)\s*)?"

_RANGE = re.compile(
    rf"\b(?:between|from)\s*{_PRICE_LOOSE}\s*(?:and|to|-)\s*{_PRICE_LOOSE}"
    rf"|{_PRICE_STRICT}\s*(?:-|to)\s*{_PRICE_STRICT}",
    re.I,
)
_MAX = re.compile(
    rf"{_FILLER}\b{_MAX_WORDS}\s*{_PRICE_LOOSE}"
    rf"|{_PRICE_STRICT}\s*(?:or less|or under|and under|or below|max)\b",
    re.I,
)
_MIN = re.compile(
    rf"{_FILLER}\b{_MIN_WORDS}\s*{_PRICE_STRICT}"
    rf"|{_PRICE_STRICT}\s*(?:\+|or more|and up|and above|or above)",
    re.I,
)
_RATING = re.compile(
    r"(?:\b(?:at least|rated|min(?:imum)?)\s*)?\b([1-5](?:\.\d)?)\s*\+?\s*stars?\b(?:\s*(?:and up|or more|or higher|\+))?"
    r"|\brated\s*([1-5](?:\.\d)?)\s*(?:\+|or (?:more|higher|above)|and up)",
    re.I,
)


# "8 x 10", "8' x 10'", "8ft by 10ft" -> size "8x10". Only room-scale numbers: "22 x 36"
# is more likely inches, and stays in the text to match titles like "22 '' x 36 ''".
_FEET_PAIR = re.compile(
    r"(?<![\d.])(\d{1,2})\s*(?:'|ft\b|feet\b)?\s*(?:x|×|by)\s*(\d{1,2})(?:\s*(?:'|ft\b|feet\b))?(?![\d.])", re.I,
)
MAX_ROOM_FEET = 15


class SearchFilters(BaseModel):
    """Structured constraints applied as Qdrant payload filters (never as text)."""

    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    categories: list[str] = Field(default_factory=list, max_length=20)
    min_rating: float | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def _check_range(self) -> "SearchFilters":
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("min_price must be <= max_price")
        return self

    def is_empty(self) -> bool:
        return (self.min_price is None and self.max_price is None
                and not self.categories and self.min_rating is None)

    def merged_over(self, base: "SearchFilters") -> "SearchFilters":
        """Fields set explicitly here win over `base` (e.g. parsed-from-text)."""
        data = base.model_dump()
        for key, value in self.model_dump().items():
            if value not in (None, []):
                data[key] = value
        if data["min_price"] is not None and data["max_price"] is not None and data["min_price"] > data["max_price"]:
            data["min_price"] = None  # explicit max wins over a conflicting parsed min
        return SearchFilters(**data)


class ParsedQuery(BaseModel):
    text: str               # query with constraint phrases removed; used for retrieval
    filters: SearchFilters
    size: str | None = None  # "8x10": a ranking preference, never a filter (see search.py)


def _to_float(whole: str, frac: str | None) -> float:
    return float(whole.replace(",", "") + (f".{frac}" if frac else ""))


def _numbers(match: re.Match) -> list[float]:
    groups = match.groups()
    return [_to_float(groups[i], groups[i + 1]) for i in range(0, len(groups) - 1, 2) if groups[i]]


def _remove(text: str, match: re.Match) -> str:
    return text[: match.start()] + " " + text[match.end():]


def parse_query(query: str) -> ParsedQuery:
    text = query
    min_price = max_price = min_rating = None

    if m := _RANGE.search(text):
        nums = _numbers(m)
        if len(nums) == 2:
            min_price, max_price = sorted(nums)
            text = _remove(text, m)
    if max_price is None and (m := _MAX.search(text)):
        max_price = _numbers(m)[0]
        text = _remove(text, m)
    if min_price is None and (m := _MIN.search(text)):
        min_price = _numbers(m)[0]
        text = _remove(text, m)
    if m := _RATING.search(text):
        min_rating = float(m.group(1) or m.group(2))
        text = _remove(text, m)

    if min_price is not None and max_price is not None and min_price > max_price:
        min_price, max_price = max_price, min_price

    cleaned = re.sub(r"\s+", " ", text).strip(" ,.-")
    cleaned = re.sub(r"\b(?:and|with|for|that|priced|costing)$", "", cleaned, flags=re.I).strip(" ,.-")
    size = None
    if (m := _FEET_PAIR.search(cleaned)) and max(int(m.group(1)), int(m.group(2))) <= MAX_ROOM_FEET:
        without = re.sub(r"\s+", " ", _remove(cleaned, m)).strip(" ,.-")
        if without:  # a query that is only a size still needs text to search with
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            size, cleaned = f"{a}x{b}", without
    return ParsedQuery(
        text=cleaned or query.strip(),
        filters=SearchFilters(min_price=min_price, max_price=max_price, min_rating=min_rating),
        size=size,
    )
