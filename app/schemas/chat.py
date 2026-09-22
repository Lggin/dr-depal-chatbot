"""Chat request/response schemas."""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class IntentType(str, Enum):
    """Intent classification."""
    KNOWLEDGE = "knowledge"
    LIVE_STATUS = "live_status"
    LIVE_DIAGNOSIS = "live_diagnosis"
    TROUBLESHOOTING = "troubleshooting"
    HISTORY = "history"
    GENERAL = "general"


class ChatRequest(BaseModel):
    """User chat request."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User message")
    language: str = Field(default="ko", description="Language code")


class DocumentSource(BaseModel):
    """Retrieved document source metadata."""
    chunk_id: str
    document: str
    section: Optional[str] = None
    alarm_code: Optional[str] = None
    trust_level: int = 100
    system: Optional[str] = None
    document_type: Optional[str] = None


class ValidationInfo(BaseModel):
    ok: bool
    is_refusal: bool
    unsupported_numbers: List[str] = []
    unsupported_codes: List[str] = []
    invalid_citations: List[str] = []
    danger_flags: List[str] = []


class RetrievalInfo(BaseModel):
    """Retrieval statistics."""
    documents_found: int
    reranked: bool
    top_k: int
    sources: List[DocumentSource]


class ChatResponse(BaseModel):
    """Chat response."""
    session_id: str
    message: str
    answer: str
    intent: IntentType
    sources: List[DocumentSource]
    tools_used: List[str] = []
    retrieval: RetrievalInfo
    validation: Optional[ValidationInfo] = None
    alarm_codes: List[str] = []
    timing_ms: dict = {}
    latency_ms: float


class ChatHistory(BaseModel):
    """Single chat turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    intent: Optional[IntentType] = None
