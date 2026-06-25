from fastembed import TextEmbedding, SparseTextEmbedding
from typing import Any


class ProductEmbedder:
    def __init__(self) -> None:
        self.dense_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    def embed_text_chunks(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, Any]]]:

        dense_vectors = list(self.dense_model.embed(texts))

        sparse_results = list(self.sparse_model.embed(texts))
        sparse_vectors = [
            {"indices": s.indices.tolist(), "values": s.values.tolist()}
            for s in sparse_results
        ]

        return dense_vectors, sparse_vectors