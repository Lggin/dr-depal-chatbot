"""Cross-encoder reranking (BAAI/bge-reranker-v2-m3) with gap-based cutoff."""

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

from app.schemas.retrieval import RetrievedDoc

logger = logging.getLogger(__name__)


def gap_cut(scores: Sequence[float], top_k: int, gap: float) -> int:
    """How many of the (desc-sorted) scores to keep.

    Walk down the ranking and stop at the first place where the score drops by
    more than `gap` from its predecessor; then cap at top_k. The gap rule runs
    FIRST so it can see past position top_k — applying the cap first once made
    the gap rule blind to anything beyond rank 6.
    """
    n = len(scores)
    if n == 0:
        return 0
    keep = n
    for i in range(1, n):
        if scores[i - 1] - scores[i] > gap:
            keep = i
            break
    return min(keep, top_k)


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 1024,
        top_k: int = 4,
        gap_cutoff: float = 3.0,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.top_k = top_k
        self.gap_cutoff = gap_cutoff
        self.model = None
        self.tokenizer = None

    def load(self) -> "Reranker":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and self.max_length > limit:
            logger.warning(f"max_length {self.max_length} > model limit {limit}; clamping")
            self.max_length = limit
        logger.info(f"✅ Reranker loaded on {self.device} in {time.perf_counter()-t0:.1f}s "
                    f"(max_length={self.max_length}, model limit={limit})")
        return self

    @staticmethod
    def _pair_text(doc: RetrievedDoc) -> str:
        """Prefix section/title metadata so the cross-encoder sees what the chunk is about."""
        m = doc.metadata
        head = " ".join(str(m[k]) for k in ("document_name", "section", "alarm_code") if m.get(k))
        return f"{head}\n{doc.text}" if head else doc.text

    def score(self, query: str, docs: List[RetrievedDoc], batch_size: int = 16) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        import torch

        out: List[float] = []
        with torch.no_grad():
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                enc = self.tokenizer(
                    [query] * len(batch), [self._pair_text(d) for d in batch],
                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.view(-1).float()
                out.extend(logits.cpu().tolist())
        return np.asarray(out)

    def rerank(
        self,
        query: str,
        docs: List[RetrievedDoc],
        top_k: Optional[int] = None,
        gap_cutoff: Optional[float] = None,
    ) -> List[RetrievedDoc]:
        if not docs:
            return []
        k = top_k or self.top_k
        gap = self.gap_cutoff if gap_cutoff is None else gap_cutoff
        t0 = time.perf_counter()

        scores = self.score(query, docs)              # score ALL candidates
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order].tolist()
        keep = gap_cut(sorted_scores, top_k=k, gap=gap)

        result = [
            docs[i].model_copy(update={"score": float(scores[i]), "rank": r, "source": "rerank"})
            for r, i in enumerate(order[:keep], start=1)
        ]
        logger.debug(f"rerank: {len(docs)} -> {keep} (gap={gap}, top_k={k}) in "
                     f"{(time.perf_counter()-t0)*1000:.0f}ms; top scores={sorted_scores[:5]}")
        return result
