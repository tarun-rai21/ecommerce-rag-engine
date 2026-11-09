"""Liveness and readiness probes, outside /api/v1 and without auth.

/healthz: the process is up (restart it if this fails).
/readyz:  it can serve searches (route traffic to it only if this succeeds). Qdrant is
          required; Redis and the LLM are reported but optional, because the API
          degrades without them (no cache; products without an answer).
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config.settings import settings

router = APIRouter(tags=["health"])


def _cache_check(result: tuple[bool, str | None]) -> dict:
    ok, error = result
    return {"ok": ok} if ok else {"ok": False, "error": error}


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    search, cache = request.app.state.search_service, request.app.state.cache_service
    try:
        points = await search.ready()
        qdrant = {"ok": points > 0, "points": points}
    except Exception as e:
        qdrant = {"ok": False, "error": type(e).__name__}
    checks = {
        "qdrant": qdrant,
        "cache": _cache_check(await cache.ping()),
        "llm": {"ok": bool(settings.GROQ_API_KEY)},  # configured; not called, to spend no quota
    }
    ready = qdrant["ok"]
    return JSONResponse({"status": "ready" if ready else "not_ready", "checks": checks},
                        status_code=200 if ready else 503)
