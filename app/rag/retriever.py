"""Retrieval pipeline orchestrator (spec §10):

    query -> [exact alarm match] + glossary expand -> dense ∥ sparse -> RRF top-20
          -> reranker (gap cut, then top-4 cap) -> exact docs re-pinned by trust
"""

import logging
import time
from typing import Any, Dict, Optional

from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.schemas.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, hybrid: HybridRetriever, reranker: Optional[Reranker] = None, pin_exact: bool = True):
        self.hybrid = hybrid
        self.reranker = reranker
        self.pin_exact = pin_exact

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        fused = self.hybrid.retrieve(query, filters=filters)
        if self.reranker is None or not fused.docs:
            return fused

        reranked = self.reranker.rerank(query, fused.docs, top_k=top_k)

        if self.pin_exact and fused.exact_chunk_ids:
            # Exact alarm-code hits are authoritative for that code: keep them
            # first, in trust order, regardless of cross-encoder opinion.
            by_id = {d.chunk_id: d for d in fused.docs}
            k = top_k or self.reranker.top_k
            pinned = [by_id[cid].model_copy(update={"source": "exact"}) for cid in fused.exact_chunk_ids][:k]
            pinned_ids = {d.chunk_id for d in pinned}
            rest = [d for d in reranked if d.chunk_id not in pinned_ids]
            reranked = pinned + rest[: max(0, k - len(pinned))]

        docs = [d.model_copy(update={"rank": r}) for r, d in enumerate(reranked, start=1)]
        return fused.model_copy(update={"docs": docs, "latency_ms": (time.perf_counter() - t0) * 1000})
