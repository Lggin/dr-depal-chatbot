"""Health check endpoints."""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "depalletizer_chatbot",
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # TODO: Check all dependencies (LLM, Qdrant, DB, etc.)
    return {
        "status": "ready",
        "services": {
            "llm": "pending",
            "qdrant": "pending",
            "database": "pending",
        }
    }
