from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.backend.api.middleware import RequestTimingMiddleware
from src.backend.api.v1.router import router as v1_router
from src.backend.services.search import ProductSearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — connecting to Qdrant.")
    service = ProductSearchService()
    app.state.search_service = service
    yield
    logger.info("Shutting down — closing Qdrant connection.")
    await service.close()


app = FastAPI(
    title="E-Commerce Search Engine",
    lifespan=lifespan,
)

app.add_middleware(RequestTimingMiddleware)
app.include_router(v1_router, prefix="/api/v1")