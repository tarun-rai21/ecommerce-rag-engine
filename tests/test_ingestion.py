import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier, SparseVectorParams, VectorParams, Distance

from src.ingestion.ingest import (
    CollectionConfigError,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    ingest_records,
)
from src.ingestion.schemas import ProductRecord


def _sparse_modifier(client: QdrantClient, name: str):
    return client.get_collection(name).config.params.sparse_vectors[SPARSE_VECTOR_NAME].modifier


def test_new_collection_uses_idf_modifier():
    client = QdrantClient(":memory:")
    ensure_collection(client, name="c")
    assert _sparse_modifier(client, "c") == Modifier.IDF


def test_collection_without_idf_is_rejected_unless_recreated():
    client = QdrantClient(":memory:")
    # Shape of the old prototype collection: sparse index without a modifier.
    client.create_collection(
        "c",
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
    )
    with pytest.raises(CollectionConfigError):
        ensure_collection(client, name="c")

    ensure_collection(client, name="c", recreate=True)
    assert _sparse_modifier(client, "c") == Modifier.IDF


def test_existing_valid_collection_is_kept():
    client = QdrantClient(":memory:")
    ensure_collection(client, name="c")
    ensure_collection(client, name="c")  # no error, no recreate
    assert _sparse_modifier(client, "c") == Modifier.IDF


def test_point_id_numeric_and_string():
    assert ProductRecord(id=101, title="x", price=1).point_id == 101
    pid = ProductRecord(id="SKU-1", title="x", price=1).point_id
    assert uuid.UUID(pid)  # valid UUID
    assert pid == ProductRecord(id="SKU-1", title="y", price=2).point_id  # deterministic


def test_ingest_isolates_bad_records_and_accepts_string_ids(embedder):
    client = QdrantClient(":memory:")
    ensure_collection(client, name="c")
    raw = [
        {"id": 1, "title": "Good numeric", "price": 10},
        {"id": "SKU-1", "title": "Good string id", "price": 5},
        {"id": 3, "title": "Negative price", "price": -1},
        {"id": 4, "title": "   ", "price": 1},
        "not a dict",
    ]
    report = ingest_records(client, embedder, raw, collection_name="c")

    assert report.loaded == 5
    assert report.uploaded == 2
    assert [idx for idx, _ in report.failures] == [2, 3, 4]
    point = client.retrieve("c", ids=[ProductRecord(id="SKU-1", title="x", price=0).point_id])[0]
    assert point.payload["product_id"] == "SKU-1"
