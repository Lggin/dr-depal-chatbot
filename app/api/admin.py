"""Admin endpoints. Ingest/reindex land with the Phase 2 corpus pipeline."""

from collections import Counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

router = APIRouter()


def _store(request: Request):
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")
    return pipe.retrieval.hybrid.store


@router.get("/admin/documents")
async def list_documents(request: Request):
    store = _store(request)
    info = await run_in_threadpool(store.get_collection_info)
    rows = await run_in_threadpool(store.scroll_all)
    by_doc = Counter(r["metadata"].get("document_name", "unknown") for r in rows)
    by_type = Counter(r["metadata"].get("document_type", "unknown") for r in rows)
    return {
        "collection": info["name"],
        "chunks": info["points_count"],
        "documents": [{"document": d, "chunks": n} for d, n in sorted(by_doc.items())],
        "by_type": dict(by_type),
    }


@router.post("/admin/ingest", status_code=501)
async def ingest():
    return {"detail": "Ingestion pipeline arrives in Phase 2. Use scripts/ingest_sample.py for now."}


@router.post("/admin/reindex", status_code=501)
async def reindex():
    return {"detail": "Reindex arrives in Phase 2. Use scripts/ingest_sample.py for now."}


@router.get("/admin/logs")
async def recent_logs(request: Request, limit: int = 20, session_id: str | None = None):
    repo = getattr(request.app.state, "log_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="Logging not available")
    rows = await run_in_threadpool(repo.recent, min(limit, 200), session_id)
    return [
        {
            "id": r.id, "timestamp": r.timestamp, "session_id": r.session_id, "question": r.question,
            "intent": r.intent, "alarm_codes": r.alarm_codes, "retrieved_chunk_ids": r.retrieved_chunk_ids,
            "latency_ms": r.latency_ms, "validation": r.validation, "user_feedback": r.user_feedback,
        }
        for r in rows
    ]
