from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    COLLECTION_NAME: str = "ecommerce_products"
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6379)

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()