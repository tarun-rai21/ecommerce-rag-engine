import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

from config.settings import settings
from src.ingestion.schemas import ProductRecord
from src.ingestion.embedder import ProductEmbedder

# ── Constants ────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "raw" / "sample_products.json"
BATCH_SIZE = 64
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 384


class CollectionConfigError(RuntimeError):
    """Existing collection was built with an incompatible configuration."""


@dataclass
class IngestionReport:
    loaded: int = 0
    uploaded: int = 0
    failures: list[tuple[int, str]] = field(default_factory=list)  # (record index, reason)


# ── Qdrant Setup ─────────────────────────────────────────
def get_client() -> QdrantClient:
    return QdrantClient(
        host=settings.QDRANT_HOST,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=True,
        timeout=60,  # collection create/delete can be slow on Docker Desktop volumes
    )


def _sparse_has_idf(client: QdrantClient, name: str) -> bool:
    params = client.get_collection(name).config.params
    sparse_cfg = (params.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)
    return sparse_cfg is not None and sparse_cfg.modifier == Modifier.IDF


def ensure_collection(
    client: QdrantClient, name: str | None = None, recreate: bool = False
) -> None:
    """Create the collection if needed.

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
    logger.info(f"Collection '{name}' created (dense cosine {DENSE_DIM}d + sparse BM25 with IDF).")


# ── Batch Processing ──────────────────────────────────────
def load_and_validate(
    raw_records: list[dict], offset: int = 0
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
                "title": record.title,
                "description": record.description,
                "price": record.price,
                "category": record.category,
                "metadata": record.metadata,
            },
        )
        for record, dense, sparse in zip(records, dense_vectors, sparse_vectors, strict=True)
    ]


def ingest_records(
    client: QdrantClient,
    embedder: ProductEmbedder,
    raw_records: list[dict],
    collection_name: str | None = None,
) -> IngestionReport:
    collection_name = collection_name or settings.COLLECTION_NAME
    report = IngestionReport(loaded=len(raw_records))

    for batch_start in range(0, len(raw_records), BATCH_SIZE):
        batch_raw = raw_records[batch_start: batch_start + BATCH_SIZE]
        logger.info(f"Processing records {batch_start} → {batch_start + len(batch_raw)}")

        valid_records, failures = load_and_validate(batch_raw, offset=batch_start)
        report.failures.extend(failures)
        if not valid_records:
            logger.warning("No valid records in this batch, skipping.")
            continue

        texts = [r.get_text_payload() for r in valid_records]
        dense_vectors, sparse_vectors = embedder.embed_documents(texts)
        points = build_points(valid_records, dense_vectors, sparse_vectors)

        client.upsert(collection_name=collection_name, points=points, wait=True)
        report.uploaded += len(points)

    return report


# ── Entry Point ───────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate, embed and index products into Qdrant.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and rebuild the collection (required once to add the BM25 IDF modifier).",
    )
    args = parser.parse_args(argv)

    logger.info("Starting ingestion pipeline.")
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_records: list[dict] = json.load(f)
    logger.info(f"Loaded {len(raw_records)} raw records from {args.data_path}.")

    client = get_client()
    ensure_collection(client, recreate=args.recreate)
    embedder = ProductEmbedder()

    report = ingest_records(client, embedder, raw_records)

    logger.info(
        f"Ingestion complete: {report.uploaded}/{report.loaded} uploaded, "
        f"{len(report.failures)} rejected."
    )
    for idx, reason in report.failures[:20]:
        logger.info(f"  rejected #{idx}: {reason}")


if __name__ == "__main__":
    main()
