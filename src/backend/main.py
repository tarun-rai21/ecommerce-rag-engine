from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings
from src.backend.api.errors import install_error_handlers
from src.backend.api.health import router as health_router
from src.backend.api.middleware import RequestContextMiddleware
from src.backend.api.v1.router import router as v1_router
from src.backend.logging_config import configure_logging
from src.backend.metrics import router as metrics_router
from src.backend.services.cache import CacheService
from src.backend.services.feedback import FeedbackStore
from src.backend.services.llm import LLMService
from src.backend.services.rerank import RerankerService
from src.backend.services.search import ProductSearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — connecting to Qdrant and Redis.")
    if not settings.api_keys:
        logger.warning("API_KEYS is empty: /api/v1 accepts requests without a key. "
                       "Fine for local development, not for a public deployment.")
    reranker = (
        RerankerService(
            settings.RERANK_MODEL,
            candidates=settings.RERANK_CANDIDATES,
            timeout_seconds=settings.RERANK_TIMEOUT_SECONDS,
        )
        if settings.RERANK_ENABLED else None
    )
    app.state.search_service = ProductSearchService(reranker=reranker)
    app.state.cache_service = CacheService()
    # Open the Redis connection now (and log why if it fails) rather than on the first request.
    if (await app.state.cache_service.ping())[0]:
        logger.info("Redis reachable.")
    app.state.llm_service = LLMService()
    app.state.feedback_store = FeedbackStore(app.state.cache_service.client,
                                            available=lambda: app.state.cache_service.available)
    yield
    logger.info("Shutting down.")
    await app.state.search_service.close()
    await app.state.cache_service.close()


configure_logging(settings.LOG_FORMAT, settings.LOG_LEVEL)

app = FastAPI(
    title="E-Commerce Search Engine",
    lifespan=lifespan,
    docs_url="/docs" if settings.DOCS_ENABLED else None,
    redoc_url="/redoc" if settings.DOCS_ENABLED else None,
    openapi_url="/openapi.json" if settings.DOCS_ENABLED else None,
)

install_error_handlers(app)
app.add_middleware(RequestContextMiddleware)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(v1_router, prefix="/api/v1")
