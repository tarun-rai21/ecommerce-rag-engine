"""Cross-encoder reranking of the top hybrid candidates.

A cross-encoder reads (query, product text) together, so it can model word
interactions that independent embeddings miss, at a much higher cost per pair.
Only the top ``candidates`` results are rescored; the rest keep their RRF order.

Safety: reranking runs in a worker thread under a time budget. On timeout or any
error the original order is returned (``reranked=False``), so a slow or broken
reranker degrades quality, never availability. Note that a timed-out thread
can't be cancelled; it finishes in the background and its result is discarded.
"""
import asyncio
from dataclasses import dataclass

from loguru import logger

from src.ingestion.schemas import build_search_text


@dataclass(frozen=True)
class RerankOutcome:
    results: list[dict]
    reranked: bool


class RerankerService:
    def __init__(
        self,
        model_name: str,
        candidates: int = 20,
        timeout_seconds: float = 2.0,
        model=None,
    ) -> None:
        if model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            model = TextCrossEncoder(model_name=model_name)
        self.model = model
        self.model_name = model_name
        self.candidates = candidates
        self.timeout_seconds = timeout_seconds
        logger.info(f"RerankerService initialized with {model_name} (top {candidates}).")

    def _score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts)]

    async def rerank(self, query: str, results: list[dict]) -> RerankOutcome:
        head, tail = results[: self.candidates], results[self.candidates:]
        if len(head) < 2:
            return RerankOutcome(results, reranked=False)
        texts = [build_search_text(r) for r in head]
        try:
            scores = await asyncio.wait_for(
                asyncio.to_thread(self._score, query, texts), timeout=self.timeout_seconds
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(f"Rerank exceeded {self.timeout_seconds:.1f}s; keeping RRF order.")
            return RerankOutcome(results, reranked=False)
        except Exception as e:
            logger.error(f"Rerank failed ({e!r}); keeping RRF order.")
            return RerankOutcome(results, reranked=False)

        order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
        reranked = [{**head[i], "rerank_score": scores[i]} for i in order]
        return RerankOutcome(reranked + tail, reranked=True)
