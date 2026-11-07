"""Loader for the Wayfair ANnotation DataSet (WANDS), MIT licence.

Source: https://github.com/wayfair/WANDS (dataset/product.csv, query.csv, label.csv;
tab-separated, UTF-8). Download with ``python -m data.download_wands``.

Produces raw dicts in the ProductRecord shape; validation happens in ingestion.
"""
import math
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.ingestion.pricing import synthetic_price

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "wands"
# Kept in the payload (and partly in the embedded text); the remaining ~400 feature
# keys are mostly compliance/warranty fields that add noise and payload size.
PAYLOAD_FEATURE_KEYS = (
    "color", "primarymaterial", "material", "framematerial", "style",
    "dsprimaryproductstyle", "shape", "upholsterycolor", "upholstered",
)
MAX_DESCRIPTION_CHARS = 2000


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)


def _float_or_none(value: Any) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def parse_features(raw: str | None) -> dict[str, str]:
    """'color : grey|overallwidth-sidetoside:64.7|…' → {'color': 'grey', …} (selected keys)."""
    features: dict[str, str] = {}
    for pair in (raw or "").split("|"):
        key, sep, value = pair.partition(":")
        key, value = key.strip().lower(), value.strip()
        if sep and key in PAYLOAD_FEATURE_KEYS and value and key not in features:
            features[key] = value
    return features


# A length written as 8 ' / 4 ' 6 '' / 1'10 '' / 4'-0 '' / 30 ''.
_LENGTH = re.compile(r"(\d+)\s*'(?!')\s*-?\s*(?:(\d+(?:\.\d+)?)\s*'')?|(\d+(?:\.\d+)?)\s*''")
_SHAPED = re.compile(r"\b(round|square|octagon|oval)\b", re.I)
BED_SIZES = {
    "twin": "twin", "twin xl": "twin xl", "extra-long twin": "twin xl", "full": "full", "full / double": "full",
    "full/double": "full", "queen": "queen", "king": "king", "california king": "california king",
}
DIMENSION_KEYS = {
    "overallwidth-sidetoside": "width", "overallheight-toptobottom": "height",
    "overalldepth-fronttoback": "depth", "overalllength-endtoend": "length", "overalllength-headtotoe": "length",
}


def _feet_inches(value: str) -> list[tuple[int, float]]:
    """"8 ' x 4 ' 6 ''" -> [(8, 0), (4, 6)]; inches-only lengths are converted to feet + inches."""
    lengths = []
    for m in _LENGTH.finditer(value):
        if m.group(3):
            total = float(m.group(3))
            lengths.append((int(total // 12), total % 12))
        else:
            lengths.append((int(m.group(1)), float(m.group(2) or 0)))
    return lengths


def _show_length(feet: int, inches: float) -> str:
    return f"{feet}'" + (f" {inches:g}\"" if inches else "")


def parse_rug_size(value: str) -> str | None:
    """"rectangle 8 ' x 10'" -> "8' x 10'"; "round 8'" -> "round 8'"; anything else -> None."""
    lengths = _feet_inches(value)
    if len(lengths) == 2:
        return " x ".join(_show_length(*length) for length in lengths)
    shape = _SHAPED.search(value)
    if len(lengths) == 1 and shape:
        return f"{shape.group(1).lower()} {_show_length(*lengths[0])}"
    return None


def _feature_values(raw: str | None) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for pair in (raw or "").split("|"):
        key, sep, value = pair.partition(":")
        if sep and value.strip():
            values.setdefault(key.strip().lower(), []).append(value.strip())
    return values


def parse_sizes(raw: str | None) -> list[str]:
    """Every purchasable size, de-duplicated in catalog order. One product can list many."""
    values = _feature_values(raw)
    sizes = [parse_rug_size(part) for v in values.get("rugsize", []) for part in v.split(",")]
    sizes += [BED_SIZES.get(v.lower()) for v in values.get("mattresssize", []) + values.get("size", [])]
    return list(dict.fromkeys(s for s in sizes if s))


def parse_dimensions(raw: str | None) -> dict[str, str]:
    """Overall dimensions in inches (checked against 3,415 titles that state a width: 94%
    agree within 0.1). Products sold in several sizes list several values, shown as a range."""
    found: dict[str, list[float]] = {}
    for key, values in _feature_values(raw).items():
        if key in DIMENSION_KEYS:
            for value in values:
                try:
                    number = float(value)
                except ValueError:
                    continue
                found.setdefault(DIMENSION_KEYS[key], []).append(number)
    return {
        name: f"{min(nums):g}" if min(nums) == max(nums) else f"{min(nums):g}-{max(nums):g}"
        for name, nums in found.items()
    }


def parse_category_path(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split("/") if part.strip()]


def iter_wands_products(data_dir: Path = DEFAULT_DIR, limit: int | None = None) -> Iterator[dict]:
    df = pd.read_csv(data_dir / "product.csv", sep="\t", encoding="utf-8", nrows=limit)
    df = df.rename(columns={"category hierarchy": "category_hierarchy"})
    for row in df.itertuples(index=False):
        product_id = str(row.product_id)
        path = parse_category_path(_clean(row.category_hierarchy))
        category = path[0] if path else "Unknown"
        description = _clean(row.product_description) or ""
        yield {
            "id": product_id,
            "title": _clean(row.product_name) or "",
            "description": description[:MAX_DESCRIPTION_CHARS],
            "price": synthetic_price(product_id, category),
            "price_is_synthetic": True,
            "category": category,
            "product_class": _clean(row.product_class),
            "category_path": path,
            "features": parse_features(_clean(row.product_features)),
            "sizes": parse_sizes(_clean(row.product_features)),
            "dimensions": parse_dimensions(_clean(row.product_features)),
            "rating": _float_or_none(row.average_rating),
            "rating_count": _int_or_none(row.rating_count),
            "review_count": _int_or_none(row.review_count),
        }
