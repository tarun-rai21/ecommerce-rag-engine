from pydantic import BaseModel, Field, field_validator
from typing import Any


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
        return str(v)

    @field_validator("title", mode="before")
    @classmethod
    def strip_and_validate_title(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("title must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("title cannot be empty")
        return stripped

    def get_text_payload(self) -> str:
        return f"Title: {self.title} | Category: {self.category} | Description: {self.description}"