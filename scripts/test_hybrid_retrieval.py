#!/usr/bin/env python3
"""Step 4 check: hybrid (exact + glossary + RRF) and hybrid+reranker on sample cases."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.dependencies import resolve_aux_device
from app.llm.embedding_client import EmbeddingClient
from app.rag import (DenseRetriever, GlossaryExpander, HybridRetriever, QdrantClient,
                     Reranker, RetrievalPipeline, SparseRetriever)
from evaluation.sample_cases import CASES, report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    device = resolve_aux_device()
    embedder = EmbeddingClient(settings.embedding_model, device=device)
    embedder.load()
    store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, embedder.embedding_dim)
    store.connect()

    dense = DenseRetriever(embedder, store, top_k=settings.dense_top_k)
    sparse = SparseRetriever.from_qdrant(store, top_k=settings.sparse_top_k)
    glossary = GlossaryExpander.from_file(settings.glossary_path)
    hybrid = HybridRetriever(dense, sparse, store, glossary,
                             settings.dense_top_k, settings.sparse_top_k, settings.fusion_top_k, settings.rrf_k)

    report("hybrid (exact+glossary+RRF, no rerank)", lambda q, f: hybrid.retrieve(q, filters=f).chunk_ids())

    reranker = Reranker(settings.reranker_model, device=device, max_length=settings.rerank_max_length,
                        top_k=settings.rerank_top_k, gap_cutoff=settings.rerank_gap_cutoff).load()
    pipeline = RetrievalPipeline(hybrid, reranker)

    lat = []
    def run(q, f):
        t0 = time.perf_counter(); r = pipeline.search(q, filters=f); lat.append((time.perf_counter() - t0) * 1000)
        return r.chunk_ids()
    report("hybrid + reranker (gap cut → top4)", run)
    print(f"latency per query: median {sorted(lat)[len(lat)//2]:.0f}ms  max {max(lat):.0f}ms  (aux device={device})")

    print("\n--- detail: VAC203이 뭐야? ---")
    r = pipeline.search("VAC203이 뭐야?")
    print(f"codes={r.alarm_codes} exact={r.exact_chunk_ids}")
    for x in r.docs:
        print(f"  {x.rank}. [{x.source:6}] {x.chunk_id:<22} trust={x.trust_level:<3} score={x.score:.3f}")
    print("\n--- detail: 빨판이 박스를 못 잡아 ---")
    r = pipeline.search("빨판이 박스를 못 잡아")
    print(f"expanded={r.expanded_query!r}")
    for x in r.docs:
        print(f"  {x.rank}. [{x.source:6}] {x.chunk_id:<22} trust={x.trust_level:<3} score={x.score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
