"""Health / readiness endpoints."""

import logging

from fastapi import APIRouter, Request, Response

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "depalletizer_chatbot", "version": settings.app_version}


@router.get("/ready")
async def readiness_check(request: Request, response: Response):
    pipe = getattr(request.app.state, "pipeline", None)
    services = {
        "llm": pipe is not None and getattr(pipe.llm, "llm", None) is not None,
        "retrieval": pipe is not None,
        "database": getattr(request.app.state, "log_repo", None) is not None,
        "qdrant": False,
    }
    if pipe is not None:
        try:
            services["qdrant"] = pipe.retrieval.hybrid.store.get_collection_info()["points_count"] >= 0
        except Exception as e:
            logger.warning(f"qdrant check failed: {e}")
    ready = all(services.values())
    response.status_code = 200 if ready else 503
    return {"status": "ready" if ready else "not_ready", "services": services}
