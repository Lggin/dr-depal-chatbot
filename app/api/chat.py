"""Chat API endpoints."""

import logging
import time
from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse, IntentType, DocumentSource, RetrievalInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Phase 1 placeholder - will be implemented in subsequent steps.
    """
    start_time = time.time()

    # TODO: Implement full chat pipeline
    # 1. Query processor
    # 2. Intent router
    # 3. Retrieval
    # 4. LLM generation
    # 5. Answer validation

    # Placeholder response
    answer = f"[PLACEHOLDER] Received: {request.message}"

    latency_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        session_id=request.session_id,
        message=request.message,
        answer=answer,
        intent=IntentType.GENERAL,
        sources=[],
        tools_used=[],
        retrieval=RetrievalInfo(
            documents_found=0,
            reranked=False,
            top_k=0,
            sources=[],
        ),
        latency_ms=latency_ms,
    )
