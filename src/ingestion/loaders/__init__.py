"""Dataset loaders: each returns an iterator of raw product dicts in ProductRecord shape."""
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from src.ingestion.loaders.wands import iter_wands_products

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = REPO_ROOT / "data" / "raw" / "sample_products.json"


def iter_sample_products(path: Path = SAMPLE_PATH, limit: int | None = None) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    yield from records[:limit] if limit else records


LOADERS: dict[str, Callable[..., Iterator[dict]]] = {
    "sample": iter_sample_products,
    "wands": iter_wands_products,
}
