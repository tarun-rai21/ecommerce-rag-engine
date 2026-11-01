"""Loader for the Wayfair ANnotation DataSet (WANDS), MIT licence.

Source: https://github.com/wayfair/WANDS (dataset/product.csv, query.csv, label.csv;
tab-separated, UTF-8). Download with ``python -m data.download_wands``.

Produces raw dicts in the ProductRecord shape; validation happens in ingestion.
"""
import math
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
            "rating": _float_or_none(row.average_rating),
            "rating_count": _int_or_none(row.rating_count),
            "review_count": _int_or_none(row.review_count),
        }
