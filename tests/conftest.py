import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.backend.main import app
from src.backend.services.cache import CacheService
from src.backend.services.feedback import FeedbackStore
from src.ingestion.embedder import ProductEmbedder
from src.ingestion.ingest import ensure_collection, ingest_records
from tests.fakes import FakeLLM, FakeRedis, FakeSearch

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "test_products"


@pytest.fixture(scope="session")
def embedder() -> ProductEmbedder:
    # Real ONNX models (downloaded once to fastembed's cache).
    return ProductEmbedder()


@pytest.fixture(scope="session")
def sample_products() -> list[dict]:
    with open(REPO_ROOT / "data" / "raw" / "sample_products.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ground_truth() -> list[dict]:
    with open(REPO_ROOT / "src" / "evaluation" / "ground_truth.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def indexed_path(tmp_path_factory, embedder, sample_products) -> str:
    """On-disk local Qdrant built with the real ingestion code. The sync client is
    closed afterwards so the async client used by the search service can open it."""
    path = str(tmp_path_factory.mktemp("qdrant"))
    client = QdrantClient(path=path)
    ensure_collection(client, name=COLLECTION)
    report = ingest_records(client, embedder, sample_products, collection_name=COLLECTION)
    assert report.uploaded == len(sample_products) and not report.failures
    client.close()
    return path


@pytest.fixture
def make_client():
    def _make(search=None, redis=None, llm=None):
        app.state.search_service = search or FakeSearch()
        app.state.cache_service = CacheService(client=redis or FakeRedis())
        app.state.feedback_store = FeedbackStore(app.state.cache_service.client,
                                                  available=lambda: app.state.cache_service.available)
        app.state.llm_service = llm or FakeLLM()
        # Not used as a context manager, so the lifespan (real services) doesn't run.
        return TestClient(app, raise_server_exceptions=False)
    return _make
