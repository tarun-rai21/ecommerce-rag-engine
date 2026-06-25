import asyncio
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)

from config.settings import settings
from src.ingestion.embedder import ProductEmbedder

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K = 60


class ProductSearchService:
    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=True,
        )
        self.embedder = ProductEmbedder()

    async def close(self) -> None:
        await self.client.close()

    # ── Vector Generation ─────────────────────────────────
    async def _vectorize_query(
        self, query: str
    ) -> tuple[list[float], dict]:
        dense_vectors, sparse_vectors = await asyncio.to_thread(
            self.embedder.embed_text_chunks, [query]
        )
        return dense_vectors[0].tolist(), sparse_vectors[0]

    # ── RRF Python Fallback ───────────────────────────────
    def _rrf_fusion(
        self,
        dense_hits: list,
        sparse_hits: list,
        limit: int,
    ) -> list[dict]:
        scores: dict[str | int, float] = {}
        payloads: dict[str | int, dict] = {}

        for rank, hit in enumerate(dense_hits):
            doc_id = hit.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            payloads[doc_id] = hit.payload

        for rank, hit in enumerate(sparse_hits):
            doc_id = hit.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            if doc_id not in payloads:
                payloads[doc_id] = hit.payload

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {"id": doc_id, "score": score, **payloads[doc_id]}
            for doc_id, score in ranked
        ]

    # ── Native Qdrant RRF ─────────────────────────────────
    async def _native_fusion_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        limit: int,
    ) -> list[dict]:
        prefetch = [
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=20,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=20,
            ),
        ]

        results = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [
            {"id": hit.id, "score": hit.score, **hit.payload}
            for hit in results.points
        ]

    # ── Fallback: parallel query_points ──────────────────
    async def _fallback_fusion_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        limit: int,
    ) -> list[dict]:
        dense_hits, sparse_hits = await asyncio.gather(
            self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=20,
                with_payload=True,
            ),
            self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=20,
                with_payload=True,
            ),
        )

        return self._rrf_fusion(
            dense_hits.points,
            sparse_hits.points,
            limit,
        )

    # ── Public Entry Point ────────────────────────────────
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        dense_vector, sparse_vector = await self._vectorize_query(query)

        try:
            results = await self._native_fusion_search(
                dense_vector, sparse_vector, limit
            )
            logger.info(f"Native RRF fusion used for query: '{query}'")
        except Exception as e:
            logger.warning(f"Native fusion failed ({e}), falling back to Python RRF.")
            results = await self._fallback_fusion_search(
                dense_vector, sparse_vector, limit
            )

        return results