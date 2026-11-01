import asyncio
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    RrfQuery,
    Rrf,
)

from config.settings import settings
from src.ingestion.embedder import ProductEmbedder

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def rrf_fuse(ranked_lists: list[list], k: int, limit: int) -> list[tuple[object, float, dict]]:
    """Reciprocal Rank Fusion: score(d) = Σ_m 1 / (k + rank_m(d)), rank starting at 1.

    Each input list holds Qdrant points (``.id``, ``.payload``) in rank order.
    Returns (point_id, score, payload) sorted by score, ties broken by first appearance.
    """
    scores: dict[object, float] = {}
    payloads: dict[object, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(hit.id, hit.payload or {})
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [(doc_id, score, payloads[doc_id]) for doc_id, score in ranked]


def _to_result(point_id: object, score: float, payload: dict) -> dict:
    # Expose the catalog's product_id; fall back to the Qdrant point id for older points.
    return {**payload, "id": payload.get("product_id", point_id), "score": score}


class ProductSearchService:
    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        embedder: ProductEmbedder | None = None,
        collection_name: str | None = None,
        rrf_k: int | None = None,
        prefetch_limit: int | None = None,
        use_native_fusion: bool | None = None,
    ) -> None:
        self.client = client or AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=True,
        )
        self.embedder = embedder or ProductEmbedder()
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.rrf_k = rrf_k or settings.RRF_K
        self.prefetch_limit = prefetch_limit or settings.PREFETCH_LIMIT
        self.use_native_fusion = (
            settings.USE_NATIVE_FUSION if use_native_fusion is None else use_native_fusion
        )

    async def close(self) -> None:
        await self.client.close()

    # ── Vector Generation ─────────────────────────────────
    async def _vectorize_query(self, query: str) -> tuple[list[float], dict]:
        # ONNX inference is CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(self.embedder.embed_query, query)

    # ── Native Qdrant RRF (one round trip) ────────────────
    async def _native_fusion_search(
        self, dense_vector: list[float], sparse_vector: SparseVector, limit: int
    ) -> list[dict]:
        results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=self.prefetch_limit),
                Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=self.prefetch_limit),
            ],
            query=RrfQuery(rrf=Rrf(k=self.rrf_k)),
            limit=limit,
            with_payload=True,
        )
        return [_to_result(hit.id, hit.score, hit.payload or {}) for hit in results.points]

    # ── Client-side RRF (two parallel queries) ────────────
    async def _python_fusion_search(
        self, dense_vector: list[float], sparse_vector: SparseVector, limit: int
    ) -> list[dict]:
        dense_hits, sparse_hits = await asyncio.gather(
            self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=self.prefetch_limit,
                with_payload=True,
            ),
            self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=self.prefetch_limit,
                with_payload=True,
            ),
        )
        fused = rrf_fuse([dense_hits.points, sparse_hits.points], k=self.rrf_k, limit=limit)
        return [_to_result(pid, score, payload) for pid, score, payload in fused]

    # ── Public Entry Point ────────────────────────────────
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        dense_vector, sparse = await self._vectorize_query(query)
        sparse_vector = SparseVector(indices=sparse["indices"], values=sparse["values"])

        if self.use_native_fusion:
            results = await self._native_fusion_search(dense_vector, sparse_vector, limit)
        else:
            results = await self._python_fusion_search(dense_vector, sparse_vector, limit)

        logger.info(
            f"Hybrid search ({'native' if self.use_native_fusion else 'python'} RRF, "
            f"k={self.rrf_k}) returned {len(results)} results for query: '{query}'"
        )
        return results
