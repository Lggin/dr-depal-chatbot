"""Dense (semantic) retrieval: query embedding -> Qdrant ANN search."""

import logging
import time
from typing import Any, Dict, Optional

from app.llm.embedding_client import EmbeddingClient
from app.rag.qdrant_client import QdrantClient
from app.schemas.retrieval import RetrievedDoc, RetrievalResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, embedder: EmbeddingClient, store: QdrantClient, top_k: int = 20):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        k = top_k or self.top_k
        t0 = time.perf_counter()

        query_vec = self.embedder.embed_single(query)
        hits = self.store.search(query_vec, top_k=k, filters=filters)

        docs = [
            RetrievedDoc(
                chunk_id=h["chunk_id"],
                text=h["text"],
                score=float(h["score"]),
                rank=i,
                source="dense",
                metadata=h["metadata"],
            )
            for i, h in enumerate(hits, start=1)
        ]
        latency = (time.perf_counter() - t0) * 1000
        logger.debug(f"dense: {len(docs)} hits in {latency:.0f}ms for {query!r} filters={filters}")
        return RetrievalResult(query=query, docs=docs, filters=filters or {}, latency_ms=latency)
