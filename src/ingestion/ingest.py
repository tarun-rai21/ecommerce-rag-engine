import json
from pathlib import Path
from loguru import logger
from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
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
DATA_PATH = Path("data/raw/sample_products.json")
BATCH_SIZE = 64
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ── Qdrant Setup ─────────────────────────────────────────
def get_client() -> QdrantClient:
    return QdrantClient(
        host=settings.QDRANT_HOST,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=True,
    )


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME in existing:
        logger.info(f"Collection '{settings.COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=384,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info(f"Collection '{settings.COLLECTION_NAME}' created.")


# ── Batch Processing ──────────────────────────────────────
def load_and_validate(raw_records: list[dict]) -> tuple[list[ProductRecord], list[int]]:
    valid_records = []
    failed_indices = []

    for idx, raw in enumerate(raw_records):
        try:
            record = ProductRecord(**raw)
            valid_records.append(record)
        except ValidationError as e:
            logger.error(f"Record #{idx} failed validation: {e.errors()}")
            failed_indices.append(idx)

    return valid_records, failed_indices


def build_points(
    records: list[ProductRecord],
    dense_vectors: list,
    sparse_vectors: list[dict],
) -> list[PointStruct]:
    points = []
    for record, dense, sparse in zip(records, dense_vectors, sparse_vectors):
        point = PointStruct(
            id=int(record.id),
            vector={
                DENSE_VECTOR_NAME: dense.tolist(),
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                ),
            },
            payload={
                "title": record.title,
                "description": record.description,
                "price": record.price,
                "category": record.category,
                "metadata": record.metadata,
            },
        )
        points.append(point)
    return points


# ── Entry Point ───────────────────────────────────────────
def main() -> None:
    logger.info("Starting ingestion pipeline.")

    # Load raw data
    with open(DATA_PATH, "r") as f:
        raw_records: list[dict] = json.load(f)
    logger.info(f"Loaded {len(raw_records)} raw records.")

    # Initialize embedder
    embedder = ProductEmbedder()

    # Initialize Qdrant
    client = get_client()
    ensure_collection(client)

    # Process in batches
    for batch_start in range(0, len(raw_records), BATCH_SIZE):
        batch_raw = raw_records[batch_start: batch_start + BATCH_SIZE]
        logger.info(f"Processing batch {batch_start} → {batch_start + len(batch_raw)}")

        valid_records, _ = load_and_validate(batch_raw)
        if not valid_records:
            logger.warning("No valid records in this batch, skipping.")
            continue

        texts = [r.get_text_payload() for r in valid_records]
        dense_vectors, sparse_vectors = embedder.embed_text_chunks(texts)

        points = build_points(valid_records, dense_vectors, sparse_vectors)

        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Uploaded {len(points)} points to Qdrant.")

    logger.info("Ingestion pipeline complete.")


if __name__ == "__main__":
    main()