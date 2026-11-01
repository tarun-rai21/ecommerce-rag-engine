import uuid
from pydantic import BaseModel, Field, field_validator
from typing import Any, Mapping

# Fixed namespace so the same product_id always maps to the same Qdrant point id.
_POINT_ID_NAMESPACE = uuid.UUID("6f1c2b1e-7a43-4c4e-9a52-1f0d3b6e8c21")

# How much of each field goes into the embedded text. Embedding cost grows with
# length (~9 docs/s on a 4-core laptop CPU at ~470 chars), so the description is
# truncated; the full description stays in the payload for the LLM.
EMBED_DESCRIPTION_CHARS = 300
EMBED_FEATURE_KEYS = ("color", "primarymaterial", "material", "style", "shape")


class ProductRecord(BaseModel):
    id: str
    title: str
    description: str = ""
    price: float = Field(ge=0.0)
    price_is_synthetic: bool = False
    category: str = "Unknown"  # top-level category, used for filtering
    product_class: str | None = None
    category_path: list[str] = []
    features: dict[str, str] = {}
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    rating_count: int | None = Field(default=None, ge=0)
    review_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = {}

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_string(cls, v: Any) -> str:
        if v is None:
            raise ValueError("id is required")
        stripped = str(v).strip()
        if not stripped:
            raise ValueError("id cannot be empty")
        return stripped

    @field_validator("title", mode="before")
    @classmethod
    def strip_and_validate_title(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("title must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("title cannot be empty")
        return stripped

    @property
    def point_id(self) -> int | str:
        """Qdrant accepts unsigned ints or UUIDs. Numeric ids are kept as ints;
        anything else (e.g. 'SKU-1') maps deterministically to a UUID."""
        if self.id.isdigit():
            return int(self.id)
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, self.id))

    def get_text_payload(self) -> str:
        """Text that gets embedded (dense) and tokenised (BM25)."""
        return build_search_text(self.model_dump())


def build_search_text(fields: Mapping[str, Any]) -> str:
    """Product text used for embedding, BM25 and reranking.

    Works on a ProductRecord dump or a Qdrant payload, so the reranker scores
    exactly the text that was indexed. Plain sentences, no field labels: labels
    like 'Title:' would be added to every document, costing tokens and adding a
    constant BM25 term."""
    title = (fields.get("title") or "").strip()
    product_class = fields.get("product_class")
    category_path = fields.get("category_path") or []
    category = fields.get("category")
    features = fields.get("features") or {}
    description = fields.get("description") or ""

    parts: list[str] = [title]
    if product_class and product_class.lower() != title.lower():
        parts.append(product_class)
    if category_path:
        parts.append(" > ".join(category_path))
    elif category and category != "Unknown":
        parts.append(category)
    attrs = [f"{k}: {features[k]}" for k in EMBED_FEATURE_KEYS if features.get(k)]
    if attrs:
        parts.append(", ".join(attrs))
    if description:
        parts.append(description[:EMBED_DESCRIPTION_CHARS])
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())
