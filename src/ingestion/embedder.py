from fastembed import TextEmbedding, SparseTextEmbedding
from typing import Any

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"


def _sparse_to_dict(embedding: Any) -> dict[str, list]:
    return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}


class ProductEmbedder:
    """Dense (BGE) + sparse (BM25) encoders.

    Documents and queries are encoded differently on purpose:
    - BM25 documents get saturated term-frequency weights; queries get weight 1.0
      per token. IDF is applied by Qdrant, which requires the sparse index to be
      created with ``modifier=Modifier.IDF`` (see ingest.ensure_collection).
    - ``query_embed`` is the model-specific query encoder for both models.
    """

    def __init__(self) -> None:
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    @property
    def model_tag(self) -> str:
        """Identifies the encoders; part of the embedding-cache key."""
        return f"{DENSE_MODEL_NAME}+{SPARSE_MODEL_NAME}"

    def embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, list]]]:
        dense_vectors = [v.tolist() for v in self.dense_model.embed(texts)]
        sparse_vectors = [_sparse_to_dict(s) for s in self.sparse_model.embed(texts)]
        return dense_vectors, sparse_vectors

    def embed_query(self, text: str) -> tuple[list[float], dict[str, list]]:
        dense = next(iter(self.dense_model.query_embed(text)))
        sparse = next(iter(self.sparse_model.query_embed(text)))
        return dense.tolist(), _sparse_to_dict(sparse)
