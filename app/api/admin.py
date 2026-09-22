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


def _reindex(request: Request, config_path: str | None):
    pipe = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded")
    lock = request.app.state.generate_lock

    def _run():
        with lock:  # no generation while the collection is rebuilt
            return pipe.reindex(config_path)
    return _run


@router.post("/admin/reindex")
async def reindex(request: Request, config_path: str | None = None):
    """Rebuild the corpus from configs/corpus.yaml (or `config_path`) into the live collection."""
    try:
        manifest = await run_in_threadpool(_reindex(request, config_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reindex failed: {type(e).__name__}: {e}")
    return {k: v for k, v in manifest.items() if k != "per_source"} | {"sources": len(manifest.get("per_source", []))}


@router.post("/admin/ingest")
async def ingest(request: Request, config_path: str | None = None):
    """Same as reindex: sources are declarative (corpus.yaml), so ingesting = rebuilding from it."""
    return await reindex(request, config_path)


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
