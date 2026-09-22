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
    foreign_script: bool = False
    fabricated_source_types: List[str] = []
    danger_flags: List[str] = []


class RetrievalInfo(BaseModel):
    """Retrieval statistics."""
    documents_found: int
    reranked: bool
    top_k: int
    sources: List[DocumentSource]
    scores: List[float] = []  # final (reranker) score per entry in `sources`


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
    retrieval_query: Optional[str] = None  # query actually sent to retrieval (after contextualization)
    route: Optional[str] = None            # which pipeline path produced the answer, e.g. "troubleshooting.ask"
    troubleshooting: Optional[dict] = None # diagnosis state snapshot when a tree is active
    machine_state: Optional[dict] = None   # live values the answer was allowed to use (from tools)
    timing_ms: dict = {}
    latency_ms: float
    log_id: Optional[int] = None


class SessionCreateResponse(BaseModel):
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    turns: int
    created_at: float
    updated_at: float


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: str = Field(..., pattern="^(up|down)$")
    note: Optional[str] = None


class ChatHistory(BaseModel):
    """Single chat turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    intent: Optional[IntentType] = None
