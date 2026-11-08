import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

from loguru import logger
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config.settings import settings
from src.ingestion.embedder import ProductEmbedder
from src.ingestion.embedding_cache import EmbeddingCache
from src.ingestion.loaders import LOADERS
from src.ingestion.schemas import ProductRecord

# ── Constants ────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
BATCH_SIZE = 64
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 384

# Payload fields that filters run on. Indexed so filtered search stays fast.
PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "category": PayloadSchemaType.KEYWORD,
    "product_class": PayloadSchemaType.KEYWORD,
    "price": PayloadSchemaType.FLOAT,
    "rating": PayloadSchemaType.FLOAT,
    "review_count": PayloadSchemaType.INTEGER,
}


class CollectionConfigError(RuntimeError):
    """Existing collection was built with an incompatible configuration."""


@dataclass
class IngestionReport:
    dataset: str = ""
    collection: str = ""
    loaded: int = 0
    uploaded: int = 0
    embedded: int = 0      # newly computed embeddings
    cache_hits: int = 0    # embeddings reused from the on-disk cache
    seconds: float = 0.0
    failures: list[tuple[int, str]] = field(default_factory=list)  # (record index, reason)


# ── Qdrant Setup ─────────────────────────────────────────
def get_client() -> QdrantClient:
    return QdrantClient(
        **settings.qdrant_client_kwargs(),
        timeout=60,  # collection create/delete can be slow on Docker Desktop volumes
    )


def _sparse_has_idf(client: QdrantClient, name: str) -> bool:
    params = client.get_collection(name).config.params
    sparse_cfg = (params.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)
    return sparse_cfg is not None and sparse_cfg.modifier == Modifier.IDF


def ensure_collection(
    client: QdrantClient, name: str | None = None, recreate: bool = False
) -> None:
    """Create the collection (with payload indexes) if needed.

    The BM25 sparse model only stores term-frequency weights; IDF is computed by
    Qdrant, and only when the sparse index has ``modifier=IDF``. That setting can't
    be changed on an existing collection, so an old collection without it is
    rejected unless ``recreate`` is set.
    """
    name = name or settings.COLLECTION_NAME

    if client.collection_exists(name):
        if recreate:
            logger.warning(f"Recreating collection '{name}' (all points will be re-uploaded).")
            client.delete_collection(name)
        elif not _sparse_has_idf(client, name):
            raise CollectionConfigError(
                f"Collection '{name}' has no IDF modifier on its sparse index, so BM25 "
                f"scores would ignore term rarity. Re-run with --recreate."
            )
        else:
            logger.info(f"Collection '{name}' already exists with a valid config.")
            return

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
                modifier=Modifier.IDF,
            )
        },
    )
    for field_name, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field_name, field_schema=schema)
    logger.info(
        f"Collection '{name}' created (dense cosine {DENSE_DIM}d + sparse BM25 with IDF; "
        f"payload indexes: {', '.join(PAYLOAD_INDEXES)})."
    )


# ── Batch Processing ──────────────────────────────────────
def load_and_validate(
    raw_records: list, offset: int = 0
) -> tuple[list[ProductRecord], list[tuple[int, str]]]:
    valid_records: list[ProductRecord] = []
    failures: list[tuple[int, str]] = []

    for i, raw in enumerate(raw_records):
        idx = offset + i
        try:
            valid_records.append(ProductRecord(**raw))
        except (ValidationError, TypeError) as e:
            reason = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            ) if isinstance(e, ValidationError) else str(e)
            logger.error(f"Record #{idx} failed validation: {reason}")
            failures.append((idx, reason))

    return valid_records, failures


def build_points(
    records: list[ProductRecord],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict],
) -> list[PointStruct]:
    return [
        PointStruct(
            id=record.point_id,
            vector={
                DENSE_VECTOR_NAME: list(dense),
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
            },
            payload={
                "product_id": record.id,
                **record.model_dump(exclude={"id"}, exclude_none=True),
            },
        )
        for record, dense, sparse in zip(records, dense_vectors, sparse_vectors, strict=True)
    ]


def embed_with_cache(
    embedder: ProductEmbedder, texts: list[str], cache: EmbeddingCache | None
) -> tuple[list[list[float]], list[dict], int]:
    """Embed texts, reusing cached vectors. Returns (dense, sparse, newly_embedded)."""
    if cache is None:
        dense, sparse = embedder.embed_documents(texts)
        return dense, sparse, len(texts)

    from src.ingestion.embedding_cache import text_key
    cached = cache.get_many(texts)
    missing = [t for t in dict.fromkeys(texts) if text_key(t) not in cached]
    if missing:
        new_dense, new_sparse = embedder.embed_documents(missing)
        cache.put_many(missing, new_dense, new_sparse)
        for t, d, s in zip(missing, new_dense, new_sparse, strict=True):
            cached[text_key(t)] = (d, s)
    vectors = [cached[text_key(t)] for t in texts]
    return [v[0] for v in vectors], [v[1] for v in vectors], len(missing)


def ingest_records(
    client: QdrantClient,
    embedder: ProductEmbedder,
    raw_records: Iterable,
    collection_name: str | None = None,
    cache: EmbeddingCache | None = None,
    total: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> IngestionReport:
    collection_name = collection_name or settings.COLLECTION_NAME
    report = IngestionReport(collection=collection_name)
    started = time.perf_counter()
    iterator = iter(raw_records)
    batch_start = 0

    while batch_raw := list(islice(iterator, batch_size)):
        report.loaded += len(batch_raw)
        valid_records, failures = load_and_validate(batch_raw, offset=batch_start)
        report.failures.extend(failures)
        batch_start += len(batch_raw)
        if not valid_records:
            logger.warning("No valid records in this batch, skipping.")
            continue

        texts = [r.get_text_payload() for r in valid_records]
        dense_vectors, sparse_vectors, n_new = embed_with_cache(embedder, texts, cache)
        report.embedded += n_new
        report.cache_hits += len(texts) - n_new

        client.upsert(
            collection_name=collection_name,
            points=build_points(valid_records, dense_vectors, sparse_vectors),
            wait=True,
        )
        report.uploaded += len(valid_records)

        elapsed = time.perf_counter() - started
        rate = report.loaded / elapsed if elapsed else 0.0
        eta = f", ETA {(total - report.loaded) / rate / 60:.1f} min" if total and rate else ""
        logger.info(
            f"{report.loaded}{f'/{total}' if total else ''} records "
            f"({report.cache_hits} cached, {report.embedded} embedded) · {rate:.1f} rec/s{eta}"
        )

    report.seconds = round(time.perf_counter() - started, 1)
    return report


def write_report(report: IngestionReport) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"ingestion_{report.dataset}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    data = asdict(report)
    data["failures"] = [{"index": i, "reason": r} for i, r in report.failures]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ── Entry Point ───────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate, embed and index products into Qdrant.")
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="wands")
    parser.add_argument("--collection", default=settings.COLLECTION_NAME)
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N products.")
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and rebuild the collection (required once to add the BM25 IDF modifier).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Don't read or write the embedding cache.")
    args = parser.parse_args(argv)

    logger.info(f"Starting ingestion: dataset={args.dataset} collection={args.collection} limit={args.limit}")
    records = list(LOADERS[args.dataset](limit=args.limit))
    logger.info(f"Loaded {len(records)} raw records.")

    client = get_client()
    ensure_collection(client, name=args.collection, recreate=args.recreate)
    embedder = ProductEmbedder()
    cache = None if args.no_cache else EmbeddingCache(model_tag=embedder.model_tag)

    try:
        report = ingest_records(
            client, embedder, records, collection_name=args.collection, cache=cache, total=len(records),
        )
    finally:
        if cache:
            cache.close()
    report.dataset = args.dataset

    logger.info(
        f"Ingestion complete in {report.seconds:.0f}s: {report.uploaded}/{report.loaded} uploaded, "
        f"{len(report.failures)} rejected, {report.cache_hits} embeddings reused from cache."
    )
    for idx, reason in report.failures[:20]:
        logger.info(f"  rejected #{idx}: {reason}")
    logger.info(f"Report written to {write_report(report)}")


if __name__ == "__main__":
    main()
