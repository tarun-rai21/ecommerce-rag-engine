from fastapi import APIRouter
from src.backend.api.v1.endpoints import router as search_router

router = APIRouter()
router.include_router(search_router)