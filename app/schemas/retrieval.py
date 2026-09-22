"""Shared retrieval result schema used by dense / sparse / hybrid / reranker."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

RetrievalSource = Literal["dense", "sparse", "hybrid", "rerank", "exact"]


class RetrievedDoc(BaseModel):
    chunk_id: str
    text: str
    score: float
    rank: int = Field(..., ge=1)
    source: RetrievalSource
    metadata: Dict[str, Any] = {}

    @property
    def alarm_code(self) -> Optional[str]:
        return self.metadata.get("alarm_code")

    @property
    def trust_level(self) -> int:
        return int(self.metadata.get("trust_level", 0))


class RetrievalResult(BaseModel):
    query: str
    docs: List[RetrievedDoc]
    filters: Dict[str, Any] = {}
    latency_ms: float = 0.0
    # Populated by the hybrid stage
    expanded_query: Optional[str] = None
    alarm_codes: List[str] = []
    exact_chunk_ids: List[str] = []

    def chunk_ids(self) -> List[str]:
        return [d.chunk_id for d in self.docs]
