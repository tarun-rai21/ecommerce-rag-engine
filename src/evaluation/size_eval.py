"""Do size queries retrieve products sold in that size?

WANDS has too few size queries to measure this (31 of 480 mention any size, and only
"carpet 5x6" is a rug size), so this uses a fixed grid of rug queries, e.g.
"grey area rug 8x10". For each query it reports how many of the top 5 results list
the requested size (read from the catalog file) and how many still match the rest
of the query.

Usage: python -m src.evaluation.size_eval
"""
import argparse
import asyncio
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import settings
from src.backend.services.query_parser import parse_query
from src.backend.services.search import ProductSearchService
from src.evaluation.retrieval_eval import MemoEmbedder
from src.evaluation.stats import bootstrap_mean, paired_difference
from src.ingestion.embedder import ProductEmbedder
from src.ingestion.loaders.wands import DEFAULT_DIR, parse_sizes
from src.ingestion.schemas import size_token

# Query -> words a suitable product should mention. Guards against the size outranking
# the rest of the query: a rejected design listed the size in 92.5% of results, but
# "grey area rug 8x10" returned espresso and green rugs.
RUGS = {
    "grey area rug": ["grey", "gray"], "blue rug": ["blue", "navy", "aqua", "teal"],
    "persian rug": ["persian", "oriental"], "shag rug": ["shag"], "outdoor rug": ["outdoor"], "wool rug": ["wool"],
    "black and white rug": ["black", "white"], "moroccan rug": ["moroccan"],
}
SIZES = ["8x10", "5x8", "9x12", "4x6", "2x3"]
TOP_K = 5


def queries() -> list[tuple[str, str]]:
    return [(f"{rug} {size}", size) for rug in RUGS for size in SIZES]


def load_size_tokens(data_dir: Path = DEFAULT_DIR) -> dict[str, set[str]]:
    df = pd.read_csv(data_dir / "product.csv", sep="\t", encoding="utf-8", usecols=["product_id", "product_features"])
    return {str(pid): {size_token(s) for s in parse_sizes(raw if isinstance(raw, str) else None)}
            for pid, raw in zip(df.product_id, df.product_features, strict=True)}


def _mentions(result: dict, words: list[str]) -> bool:
    text = " ".join(str(result.get(k, "")) for k in ("title", "description", "features")).lower()
    return any(w in text for w in words)


def _is_rug(result: dict) -> bool:
    return "rug" in f"{result.get('product_class', '')} {result.get('title', '')}".lower()


async def evaluate(service: ProductSearchService, tokens: dict[str, set[str]], size_aware: bool) -> dict[str, list[float]]:
    """size_aware: the production path. Otherwise the whole query is plain search text."""
    rows: dict[str, list[float]] = {"size": [], "any": [], "term": [], "rug": []}
    for text, size in queries():
        base = text.rsplit(" ", 1)[0]
        parsed = parse_query(text)
        results = (await service.search(parsed.text, limit=TOP_K, prefer_size=parsed.size) if size_aware
                   else await service.search(text, limit=TOP_K))
        offering = sum(size in tokens.get(str(r["id"]), set()) for r in results)
        rows["size"].append(offering / TOP_K)
        rows["any"].append(float(offering > 0))
        rows["term"].append(sum(_mentions(r, RUGS[base]) for r in results) / TOP_K)
        rows["rug"].append(sum(_is_rug(r) for r in results) / TOP_K)
    return rows


async def run(collection: str) -> None:
    tokens = load_size_tokens()
    service = ProductSearchService(embedder=MemoEmbedder(ProductEmbedder()), collection_name=collection)
    try:
        plain = await evaluate(service, tokens, size_aware=False)
        aware = await evaluate(service, tokens, size_aware=True)
    finally:
        await service.close()
    print(f"\n{len(queries())} queries; top {TOP_K} results; collection {collection}.\n")
    print("| System | Lists the size | Queries with at least one | Mentions the requested term | Is a rug |")
    print("|---|---|---|---|---|")
    for name, r in (("Size as search text", plain), ("Size as preference", aware)):
        print(f"| {name} | " + " | ".join(bootstrap_mean(r[k]).fmt() for k in ("size", "any", "term", "rug")) + " |")
    print(f"\nLists the size, difference: {paired_difference(aware['size'], plain['size']).fmt()}")
    print(f"Mentions the term, difference: {paired_difference(aware['term'], plain['term']).fmt()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=settings.COLLECTION_NAME)
    logger.remove()
    asyncio.run(run(parser.parse_args().collection))


if __name__ == "__main__":
    main()
