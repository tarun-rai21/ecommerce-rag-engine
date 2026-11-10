"""Versioned collections behind an alias: build, validate, switch, prune, roll back."""
import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from src.backend.services.search import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ProductSearchService
from src.ingestion.ingest import ensure_collection, ingest_records
from src.ingestion.versioning import (
    PromotionError,
    alias_target,
    next_version,
    promote,
    rollback,
    validate,
    versioned_name,
    versions,
)

ALIAS = "products"


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def build(client, embedder, records, version: int) -> str:
    name = versioned_name(ALIAS, version)
    ensure_collection(client, name=name)
    report = ingest_records(client, embedder, records, collection_name=name)
    assert report.uploaded == len(records)
    return name


def test_build_switch_prune_and_rollback(client, embedder, sample_products):
    records = sample_products[:12]
    assert next_version(client, ALIAS) == 1
    v1 = build(client, embedder, records, 1)
    assert validate(client, v1, len(records), embedder, SPARSE_VECTOR_NAME, DENSE_VECTOR_NAME) == []
    assert promote(client, ALIAS, v1) is None and alias_target(client, ALIAS) == v1

    v2 = build(client, embedder, records, next_version(client, ALIAS))
    assert promote(client, ALIAS, v2) == v1 and alias_target(client, ALIAS) == v2
    assert client.get_collection(ALIAS).points_count == len(records)  # reads go through the alias

    v3 = build(client, embedder, records, next_version(client, ALIAS))
    promote(client, ALIAS, v3, keep=2)
    assert list(versions(client, ALIAS).values()) == [v2, v3]  # v1 pruned; v2 kept for rollback

    assert rollback(client, ALIAS) == v2 and alias_target(client, ALIAS) == v2
    with pytest.raises(PromotionError):
        rollback(client, ALIAS)  # nothing older than v2 is left


def test_validation_blocks_a_bad_build(client, embedder, sample_products):
    v1 = build(client, embedder, sample_products[:5], 1)
    problems = validate(client, v1, 50, embedder, SPARSE_VECTOR_NAME, DENSE_VECTOR_NAME)
    assert problems == ["5 points, expected 50"]


def test_plain_collection_is_only_replaced_when_asked(client, embedder, sample_products):
    ensure_collection(client, name=ALIAS)  # an old, unversioned index under the alias's name
    v1 = build(client, embedder, sample_products[:3], 1)
    with pytest.raises(PromotionError, match="plain collection"):
        promote(client, ALIAS, v1)
    assert client.collection_exists(ALIAS) and alias_target(client, ALIAS) is None  # untouched
    promote(client, ALIAS, v1, replace_collection=True)
    assert alias_target(client, ALIAS) == v1


def test_search_service_resolves_the_alias_for_cache_keys():
    class FakeQdrant:
        def __init__(self):
            self.target, self.calls = "products_v1", 0

        async def get_aliases(self):
            self.calls += 1
            return SimpleNamespace(aliases=[SimpleNamespace(alias_name="products", collection_name=self.target)])

    fake = FakeQdrant()
    service = ProductSearchService(client=fake, embedder=object(), collection_name="products")
    assert asyncio.run(service.active_collection()) == "products_v1"
    fake.target = "products_v2"
    assert asyncio.run(service.active_collection()) == "products_v1"  # memoised for 30 s
    service._active_at -= ProductSearchService.ACTIVE_TTL_SECONDS
    assert asyncio.run(service.active_collection()) == "products_v2" and fake.calls == 2

    plain = ProductSearchService(client=FakeQdrant(), embedder=object(), collection_name="unaliased")
    assert asyncio.run(plain.active_collection()) == "unaliased"  # not an alias: its own name


def test_failed_build_is_deleted_so_rollback_cannot_reach_it(client, embedder, sample_products, monkeypatch, tmp_path):
    # A build that fails validation used to stay as "<alias>_vN"; --rollback picks the newest
    # version below the current one, so it could later switch to that never-validated build.
    from src.ingestion import ingest

    v1 = build(client, embedder, sample_products[:5], 1)
    promote(client, ALIAS, v1)
    monkeypatch.setattr(ingest, "get_client", lambda: client)
    monkeypatch.setattr(ingest, "ProductEmbedder", lambda: embedder)
    monkeypatch.setattr(ingest, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "LOADERS", {"sample": lambda limit=None: sample_products[:5]})
    monkeypatch.setattr(ingest, "validate", lambda *a, **kw: ["forced failure"])
    with pytest.raises(SystemExit):
        ingest.main(["--dataset", "sample", "--collection", ALIAS, "--versioned", "--no-cache"])
    assert list(versions(client, ALIAS).values()) == [v1]  # v2 was deleted
    assert alias_target(client, ALIAS) == v1
