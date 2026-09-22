"""Chat and feedback endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.schemas.chat import ChatRequest, ChatResponse, FeedbackRequest

logger = logging.getLogger(__name__)

router = APIRouter()


def _pipeline(request: Request):
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")
    return pipe


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    pipe = _pipeline(request)
    lock = request.app.state.generate_lock  # llama.cpp context is single-threaded

    def _run():
        with lock:
            return pipe.answer(body.session_id, body.message)

    try:
        response: ChatResponse = await run_in_threadpool(_run)
    except Exception as e:  # keep the server alive; surface the reason
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=f"generation failed: {type(e).__name__}: {e}")

    repo = getattr(request.app.state, "log_repo", None)
    if repo is not None:
        try:
            response.log_id = await run_in_threadpool(repo.log, response)
        except Exception:
            logger.exception("chat log write failed")
    return response


@router.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request):
    repo = getattr(request.app.state, "log_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="Logging not available")
    ok = await run_in_threadpool(repo.set_feedback, body.log_id, body.feedback, body.note)
    if not ok:
        raise HTTPException(status_code=404, detail=f"log_id {body.log_id} not found")
    return {"ok": True, "log_id": body.log_id, "feedback": body.feedback}
