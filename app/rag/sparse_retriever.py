"""Sparse (lexical) retrieval: BM25 over Kiwi tokens, with metadata post-filtering."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.tokenizer import KoreanTokenizer
from app.schemas.retrieval import RetrievedDoc, RetrievalResult

logger = logging.getLogger(__name__)


def _matches(meta: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Same semantics as metadata_filter.build_filter, applied in Python."""
    if not filters:
        return True
    for key, cond in filters.items():
        if cond is None:
            continue
        val = meta.get(key)
        if isinstance(cond, dict):
            if val is None:
                return False
            if "gte" in cond and val < cond["gte"]: return False
            if "gt" in cond and val <= cond["gt"]: return False
            if "lte" in cond and val > cond["lte"]: return False
            if "lt" in cond and val >= cond["lt"]: return False
        elif isinstance(cond, (list, tuple, set)):
            if val not in cond:
                return False
        elif val != cond:
            return False
    return True


class SparseRetriever:
    def __init__(self, tokenizer: Optional[KoreanTokenizer] = None, top_k: int = 20):
        self.tokenizer = tokenizer or KoreanTokenizer()
        self.top_k = top_k
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> "SparseRetriever":
        """chunks: [{chunk_id, text, metadata}] — same shape QdrantClient returns."""
        t0 = time.perf_counter()
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._texts = [c["text"] for c in chunks]
        self._metas = [c.get("metadata", {}) for c in chunks]
        # Index heading-prefixed search_text when present so section titles are matchable.
        index_texts = [m.get("search_text") or t for t, m in zip(self._texts, self._metas)]
        self._bm25 = BM25Okapi([self.tokenizer.tokenize(t) for t in index_texts])
        logger.info(f"BM25 index: {len(chunks)} chunks in {(time.perf_counter()-t0)*1000:.0f}ms")
        return self

    @classmethod
    def from_qdrant(cls, store, **kwargs) -> "SparseRetriever":
        """Build from the vector store so both retrievers see exactly the same chunks."""
        return cls(**kwargs).build(store.scroll_all())

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() or from_qdrant() first.")
        k = top_k or self.top_k
        t0 = time.perf_counter()

        q_tokens = self.tokenizer.tokenize(query)
        scores = np.asarray(self._bm25.get_scores(q_tokens))

        candidates = [
            i for i in np.argsort(-scores)
            if scores[i] > 0 and _matches(self._metas[i], filters)
        ][:k]

        docs = [
            RetrievedDoc(
                chunk_id=self._chunk_ids[i],
                text=self._texts[i],
                score=float(scores[i]),
                rank=r,
                source="sparse",
                metadata=self._metas[i],
            )
            for r, i in enumerate(candidates, start=1)
        ]
        latency = (time.perf_counter() - t0) * 1000
        logger.debug(f"sparse: {len(docs)} hits in {latency:.0f}ms tokens={q_tokens}")
        return RetrievalResult(query=query, docs=docs, filters=filters or {}, latency_ms=latency)
