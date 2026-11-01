import uuid
from pydantic import BaseModel, Field, field_validator
from typing import Any

# Fixed namespace so the same product_id always maps to the same Qdrant point id.
_POINT_ID_NAMESPACE = uuid.UUID("6f1c2b1e-7a43-4c4e-9a52-1f0d3b6e8c21")


class ProductRecord(BaseModel):
    id: str
    title: str
    description: str = ""
    price: float = Field(ge=0.0)
    category: str = "Unknown"
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
        return f"Title: {self.title} | Category: {self.category} | Description: {self.description}"
