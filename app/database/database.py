"""SQLAlchemy engine/session setup (SQLite by default)."""

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
_SessionFactory = None


def init_db(database_url: str):
    """Create engine + tables. ':memory:' URLs share one connection (tests)."""
    global _engine, _SessionFactory
    from app.database import models  # noqa: F401  (register tables)

    kwargs = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
        else:
            path = database_url.split("sqlite:///", 1)[-1]
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(database_url, future=True, **kwargs)
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    logger.info(f"database ready: {database_url}")
    return _SessionFactory


def get_session_factory():
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _SessionFactory
