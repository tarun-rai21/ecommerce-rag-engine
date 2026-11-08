import asyncio
from typing import Literal

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    Prefetch,
    Range,
    Rrf,
    RrfQuery,
    SparseVector,
)

from config.settings import settings
from src.backend.services.query_parser import SearchFilters
from src.backend.services.rerank import RerankerService
from src.ingestion.embedder import ProductEmbedder
from src.ingestion.schemas import size_token

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SearchMode = Literal["hybrid", "hybrid_rerank", "dense", "sparse"]
# A size preference re-orders this many of the query's best matches. Chosen with size_eval:
# the deepest setting where results still matching the rest of the query fall at most 2
# points (15: -0.5). Deeper finds more sizes but loses relevance (100: -8.5).
SIZE_CANDIDATES = 15


def size_first(results: list[dict], size: str, limit: int) -> list[dict]:
    """Products sold in `size` first, otherwise keeping relevance order.

    A preference, not a filter: 60% of rugs list no size, and only the query's best
    matches are re-ordered, so "8x10 picture frame" can't turn into 8x10 rugs."""
    has_size = [r for r in results if size in {size_token(s) for s in r.get("sizes") or []}]
    matched = {id(r) for r in has_size}
    return (has_size + [r for r in results if id(r) not in matched])[:limit]


def rrf_fuse(ranked_lists: list[list], k: int, limit: int) -> list[tuple[object, float, dict]]:
    """Reciprocal Rank Fusion: score(d) = Σ_m 1 / (k + rank_m(d)), rank starting at 1.

    Each input list holds Qdrant points (``.id``, ``.payload``) in rank order.
    Returns (point_id, score, payload) sorted by score, ties broken by id.
    """
    scores: dict[object, float] = {}
    payloads: dict[object, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(hit.id, hit.payload or {})
    ranked = sorted(scores.items(), key=lambda x: _rank_key(x[0], x[1]))[:limit]
    return [(doc_id, score, payloads[doc_id]) for doc_id, score in ranked]


# RRF produces exact ties often: a product ranked r-th by dense only scores the same
# as one ranked r-th by BM25 only. Qdrant returns tied points in an order that varies
# between runs, so both fusion paths sort by (score, id). Scores are rounded because
# Qdrant computes them in float32 and rrf_fuse in float64.
TIE_SLACK = 10  # extra native results fetched so a tie at the cut-off can't change the set


def _rank_key(point_id: object, score: float) -> tuple[float, str]:
    return (-round(score, 6), str(point_id))


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
        reranker: RerankerService | None = None,
    ) -> None:
        self.client = client or AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=True,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
        )
        self.embedder = embedder or ProductEmbedder()
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.rrf_k = rrf_k or settings.RRF_K
        self.prefetch_limit = prefetch_limit or settings.PREFETCH_LIMIT
        self.use_native_fusion = (
            settings.USE_NATIVE_FUSION if use_native_fusion is None else use_native_fusion
        )
        self.reranker = reranker

    def _prefetch(self, limit: int) -> int:
        # Each fusion leg needs at least `limit` candidates to fill the result list.
        return max(self.prefetch_limit, limit)

    async def close(self) -> None:
        await self.client.close()

    async def ready(self) -> int:
        """Points in the collection; raises if Qdrant or the collection is unreachable."""
        info = await asyncio.wait_for(self.client.get_collection(self.collection_name), timeout=2.0)
        return info.points_count or 0

    # ── Vector Generation ─────────────────────────────────
    async def _vectorize_query(self, query: str) -> tuple[list[float], dict]:
        # ONNX inference is CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(self.embedder.embed_query, query)

    # ── Native Qdrant RRF (one round trip) ────────────────
    async def _native_fusion_search(
        self, dense_vector: list[float], sparse_vector: SparseVector, limit: int,
        qfilter: Filter | None = None,
    ) -> list[dict]:
        # The filter goes inside each prefetch so every leg returns `prefetch` matching
        # candidates; filtering only after fusion could leave fewer than `limit` results.
        results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME,
                         limit=self._prefetch(limit), filter=qfilter),
                Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME,
                         limit=self._prefetch(limit), filter=qfilter),
            ],
            query=RrfQuery(rrf=Rrf(k=self.rrf_k)),
            query_filter=qfilter,
            limit=limit + TIE_SLACK,
            with_payload=True,
        )
        hits = sorted(results.points, key=lambda h: _rank_key(h.id, h.score))[:limit]
        return [_to_result(hit.id, hit.score, hit.payload or {}) for hit in hits]

    # ── Client-side RRF (two parallel queries) ────────────
    async def _python_fusion_search(
        self, dense_vector: list[float], sparse_vector: SparseVector, limit: int,
        qfilter: Filter | None = None,
    ) -> list[dict]:
        dense_hits, sparse_hits = await asyncio.gather(
            self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=self._prefetch(limit),
                with_payload=True,
            ),
            self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=self._prefetch(limit),
                with_payload=True,
            ),
        )
        fused = rrf_fuse([dense_hits.points, sparse_hits.points], k=self.rrf_k, limit=limit)
        return [_to_result(pid, score, payload) for pid, score, payload in fused]

    # ── Single-retriever search (baselines) ───────────────
    async def _single_search(self, vector, using: str, limit: int, qfilter: Filter | None = None) -> list[dict]:
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=using,
            query_filter=qfilter,
            limit=limit + TIE_SLACK,
            with_payload=True,
        )
        # BM25 scores tie often (same matched terms, same length); same rule as fusion.
        hits = sorted(results.points, key=lambda h: (-h.score, str(h.id)))[:limit]
        return [_to_result(hit.id, hit.score, hit.payload or {}) for hit in hits]

    # ── Public Entry Point ────────────────────────────────
    async def search(
        self,
        query: str,
        limit: int = 5,
        mode: SearchMode | None = None,
        filters: SearchFilters | None = None,
        prefer_size: str | None = None,
    ) -> list[dict]:
        """mode: 'hybrid' (dense + BM25 fused with RRF), 'hybrid_rerank' (hybrid, then
        a cross-encoder reorders the top candidates), or 'dense' / 'sparse' alone
        (evaluation baselines). Default: 'hybrid_rerank' if a reranker is configured,
        else 'hybrid'.

        filters: hard constraints applied as Qdrant payload filters (every result
        satisfies them). Parse them from text first with query_parser.parse_query.

        prefer_size: e.g. "8x10"; products sold in that size rank first (see size_first)."""
        if prefer_size:
            wider = await self.search(query, max(limit, SIZE_CANDIDATES), mode, filters)
            return size_first(wider, prefer_size, limit)
        mode = mode or ("hybrid_rerank" if self.reranker else "hybrid")
        if mode not in ("hybrid", "hybrid_rerank", "dense", "sparse"):
            raise ValueError(f"Unknown search mode: {mode!r}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' needs a reranker")
        qfilter = build_qdrant_filter(filters)

        dense_vector, sparse = await self._vectorize_query(query)
        sparse_vector = SparseVector(indices=sparse["indices"], values=sparse["values"])

        if mode == "dense":
            results = await self._single_search(dense_vector, DENSE_VECTOR_NAME, limit, qfilter)
        elif mode == "sparse":
            results = await self._single_search(sparse_vector, SPARSE_VECTOR_NAME, limit, qfilter)
        else:
            fetch = max(limit, self.reranker.candidates) if mode == "hybrid_rerank" else limit
            if self.use_native_fusion:
                results = await self._native_fusion_search(dense_vector, sparse_vector, fetch, qfilter)
            else:
                results = await self._python_fusion_search(dense_vector, sparse_vector, fetch, qfilter)
            if mode == "hybrid_rerank":
                results = (await self.reranker.rerank(query, results)).results[:limit]

        logger.debug(f"{mode} search returned {len(results)} results for query: '{query}'")
        return results

    # ── Facets (for filter UIs) ───────────────────────────
    async def facet_counts(self, key: str = "category", limit: int = 100) -> list[dict]:
        """Distinct payload values with counts, e.g. categories for a filter dropdown."""
        response = await self.client.facet(collection_name=self.collection_name, key=key, limit=limit)
        return [{"value": hit.value, "count": hit.count} for hit in response.hits]


def build_qdrant_filter(filters: SearchFilters | None) -> Filter | None:
    """SearchFilters → Qdrant Filter. Products with no rating never match a rating
    filter (an unknown rating isn't evidence of a good one)."""
    if filters is None or filters.is_empty():
        return None
    must: list[FieldCondition] = []
    if filters.min_price is not None or filters.max_price is not None:
        must.append(FieldCondition(key="price", range=Range(gte=filters.min_price, lte=filters.max_price)))
    if filters.categories:
        must.append(FieldCondition(key="category", match=MatchAny(any=filters.categories)))
    if filters.min_rating is not None:
        must.append(FieldCondition(key="rating", range=Range(gte=filters.min_rating)))
    return Filter(must=must)
