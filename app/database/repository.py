"""Persistence for chat logs."""

from typing import List, Optional

from sqlalchemy import select

from app.database.database import get_session_factory
from app.database.models import ChatLog
from app.schemas.chat import ChatResponse


class ChatLogRepository:
    def __init__(self, session_factory=None):
        self._sf = session_factory

    @property
    def sf(self):
        return self._sf or get_session_factory()

    def log(self, r: ChatResponse) -> int:
        row = ChatLog(
            session_id=r.session_id,
            question=r.message,
            normalized_query=r.retrieval_query,
            intent=str(r.intent.value if hasattr(r.intent, "value") else r.intent),
            alarm_codes=r.alarm_codes,
            retrieved_chunk_ids=[s.chunk_id for s in r.retrieval.sources],
            reranker_scores=r.retrieval.scores,
            sources=[s.model_dump() for s in r.sources],
            tools_called=r.tools_used,
            answer=r.answer,
            validation=r.validation.model_dump() if r.validation else {},
            timing_ms=r.timing_ms,
            latency_ms=r.latency_ms,
        )
        with self.sf() as s:
            s.add(row)
            s.commit()
            return row.id

    def set_feedback(self, log_id: int, feedback: str, note: Optional[str] = None) -> bool:
        with self.sf() as s:
            row = s.get(ChatLog, log_id)
            if row is None:
                return False
            row.user_feedback = feedback
            row.feedback_note = note
            s.commit()
            return True

    def recent(self, limit: int = 50, session_id: Optional[str] = None) -> List[ChatLog]:
        stmt = select(ChatLog).order_by(ChatLog.id.desc()).limit(limit)
        if session_id:
            stmt = stmt.where(ChatLog.session_id == session_id)
        with self.sf() as s:
            return list(s.scalars(stmt))

    def count(self) -> int:
        with self.sf() as s:
            return s.query(ChatLog).count()
