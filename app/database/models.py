"""Chat log table (spec §25). Only what evaluation and future SFT need — no raw machine data dumps."""

import time
from typing import Optional

from sqlalchemy import JSON, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[float] = mapped_column(Float, default=time.time, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)

    question: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    intent: Mapped[str] = mapped_column(String(32))
    alarm_codes: Mapped[list] = mapped_column(JSON, default=list)

    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    reranker_scores: Mapped[list] = mapped_column(JSON, default=list)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    tools_called: Mapped[list] = mapped_column(JSON, default=list)

    answer: Mapped[str] = mapped_column(Text)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    timing_ms: Mapped[dict] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float] = mapped_column(Float)

    user_feedback: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # "up" | "down"
    feedback_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
