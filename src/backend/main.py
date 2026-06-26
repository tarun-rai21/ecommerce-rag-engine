from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.backend.api.middleware import RequestTimingMiddleware
from src.backend.api.v1.router import router as v1_router
from src.backend.services.search import ProductSearchService
from src.backend.services.cache import CacheService
from src.backend.services.llm import LLMService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — connecting to Qdrant and Redis.")
    app.state.search_service = ProductSearchService()
    app.state.cache_service = CacheService()
    app.state.llm_service = LLMService()
    yield
    logger.info("Shutting down.")
    await app.state.search_service.close()
    await app.state.cache_service.close()


app = FastAPI(
    title="E-Commerce Search Engine",
    lifespan=lifespan,
)

app.add_middleware(RequestTimingMiddleware)
app.include_router(v1_router, prefix="/api/v1")