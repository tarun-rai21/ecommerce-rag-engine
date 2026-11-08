from fastapi import APIRouter, Depends

from src.backend.api.security import require_api_key
from src.backend.api.v1.endpoints import router as search_router

router = APIRouter(dependencies=[Depends(require_api_key)])
router.include_router(search_router)
