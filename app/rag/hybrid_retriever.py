"""Hybrid retrieval: exact alarm-code match, then RRF fusion of dense + sparse.

Priority (spec §7):
  1. exact alarm_code match  -> pinned first, ordered by trust_level desc
  2. metadata-filtered dense ∥ sparse -> Reciprocal Rank Fusion
"""

import logging
import time
from typing import Any, Dict, List, Optional

from app.rag.dense_retriever import DenseRetriever
from app.rag.query_expander import GlossaryExpander
from app.rag.sparse_retriever import SparseRetriever
from app.router.alarm_parser import extract_alarm_codes
from app.schemas.retrieval import RetrievedDoc, RetrievalResult

logger = logging.getLogger(__name__)


def rrf_fuse(rankings: List[List[RetrievedDoc]], k: int = 60) -> List[RetrievedDoc]:
    """Reciprocal Rank Fusion. score = Σ 1/(k + rank) over lists the doc appears in."""
    scores: Dict[str, float] = {}
    docs: Dict[str, RetrievedDoc] = {}
    for ranking in rankings:
        for doc in ranking:
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0.0) + 1.0 / (k + doc.rank)
            docs.setdefault(doc.chunk_id, doc)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        docs[cid].model_copy(update={"score": s, "rank": r, "source": "hybrid"})
        for r, (cid, s) in enumerate(ordered, start=1)
    ]


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        store,
        expander: Optional[GlossaryExpander] = None,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        fusion_top_k: int = 20,
        rrf_k: int = 60,
        pin_min_trust: int = 85,
    ):
        self.dense = dense
        self.sparse = sparse
        self.store = store
        self.expander = expander
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_k = fusion_top_k
        self.rrf_k = rrf_k
        self.pin_min_trust = pin_min_trust

    def _exact(self, codes: List[str], filters: Optional[Dict[str, Any]]) -> List[RetrievedDoc]:
        rows = []
        for code in codes:
            rows.extend(self.store.scroll_by_filter({**(filters or {}), "alarm_code": code}))
        rows.sort(key=lambda h: (-int(h["metadata"].get("trust_level", 0)), h["chunk_id"]))
        return [
            RetrievedDoc(chunk_id=h["chunk_id"], text=h["text"], score=1.0, rank=r,
                         source="exact", metadata=h["metadata"])
            for r, h in enumerate(rows, start=1)
        ]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        k = top_k or self.fusion_top_k
        t0 = time.perf_counter()

        codes = extract_alarm_codes(query)
        exact = self._exact(codes, filters) if codes else []

        expanded, matched = self.expander.expand_with_terms(query) if self.expander else (query, [])
        dense_docs = self.dense.retrieve(expanded, top_k=self.dense_top_k, filters=filters).docs
        sparse_docs = self.sparse.retrieve(expanded, top_k=self.sparse_top_k, filters=filters).docs
        fused = rrf_fuse([dense_docs, sparse_docs], k=self.rrf_k)

        pinned = [d for d in exact if d.trust_level >= self.pin_min_trust]
        unpinned = [d for d in exact if d.trust_level < self.pin_min_trust]
        seen = {d.chunk_id for d in pinned}
        merged = list(pinned)
        for d in fused + unpinned:  # low-trust exact hits join the candidate pool for the reranker
            if d.chunk_id not in seen:
                merged.append(d); seen.add(d.chunk_id)
        docs = [d.model_copy(update={"rank": r}) for r, d in enumerate(merged[:k], start=1)]

        latency = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"hybrid: codes={codes} exact={len(exact)} pinned={len(pinned)} glossary={matched} "
            f"dense={len(dense_docs)} sparse={len(sparse_docs)} -> {len(docs)} in {latency:.0f}ms"
        )
        return RetrievalResult(
            query=query, docs=docs, filters=filters or {}, latency_ms=latency,
            expanded_query=expanded if expanded != query else None,
            alarm_codes=codes, exact_chunk_ids=[d.chunk_id for d in pinned],
        )
