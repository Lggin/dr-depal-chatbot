"""FastAPI application entry point."""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, chat, health, session
from app.config import ensure_directories, settings

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version} ({settings.environment})")
    ensure_directories()
    app.state.generate_lock = threading.Lock()
    app.state.pipeline = None
    app.state.log_repo = None

    from app.database import ChatLogRepository, init_db
    init_db(settings.database_url)
    app.state.log_repo = ChatLogRepository()

    if settings.skip_model_load:
        logger.warning("SKIP_MODEL_LOAD=1 — API up without models (/api/chat will return 503)")
    else:
        from app.dependencies import build_chat_pipeline
        app.state.pipeline = build_chat_pipeline()

    yield

    pipe = getattr(app.state, "pipeline", None)
    if pipe is not None:
        pipe.close()
    logger.info("Shutdown complete")


app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(session.router, prefix="/api", tags=["session"])
app.include_router(admin.router, prefix="/api", tags=["admin"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
