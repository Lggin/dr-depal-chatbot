"""Session lifecycle endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import SessionCreateResponse, SessionInfo

router = APIRouter()


def _store(request: Request):
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")
    return pipe.sessions


@router.post("/session", response_model=SessionCreateResponse, status_code=201)
async def create_session(request: Request):
    sid = uuid.uuid4().hex[:12]
    _store(request).get_or_create(sid)
    return SessionCreateResponse(session_id=sid)


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, request: Request):
    s = _store(request).get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionInfo(session_id=s.session_id, turns=s.turns, created_at=s.created_at, updated_at=s.updated_at)


@router.delete("/session/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    if not _store(request).delete(session_id):
        raise HTTPException(status_code=404, detail="session not found")
