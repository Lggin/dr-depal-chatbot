#!/usr/bin/env python3
"""Step 6 check: retrieval → prompt → LLM on a few queries, incl. one outside the corpus.

Loads embedding + reranker (aux device) and the LLM (GPU) together — also a first
co-residency check for VRAM.
"""

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.dependencies import resolve_aux_device
from app.llm.client import LlamaCppClient
from app.llm.embedding_client import EmbeddingClient
from app.llm.prompt import ContextBuilder, PromptBuilder
from app.rag import (DenseRetriever, GlossaryExpander, HybridRetriever, QdrantClient,
                     Reranker, RetrievalPipeline, SparseRetriever)

logging.basicConfig(level=logging.WARNING)

QUERIES = [
    "VAC203이 뭐야?",
    "빨판이 박스를 못 잡아. 뭘 확인해야 해?",
    "VAC-203 지난번에도 있었어?",
    "로봇 컨트롤러 IP 주소가 뭐야?",   # not in corpus → expect explicit "확인할 수 없습니다"
]


def vram():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip()
    except Exception:
        return "n/a"


def main() -> int:
    queries = sys.argv[1:] or QUERIES
    device = resolve_aux_device()
    embedder = EmbeddingClient(settings.embedding_model, device=device); embedder.load()
    store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, embedder.embedding_dim); store.connect()
    dense = DenseRetriever(embedder, store, top_k=settings.dense_top_k)
    sparse = SparseRetriever.from_qdrant(store, top_k=settings.sparse_top_k)
    hybrid = HybridRetriever(dense, sparse, store, GlossaryExpander.from_file(settings.glossary_path),
                             settings.dense_top_k, settings.sparse_top_k, settings.fusion_top_k, settings.rrf_k)
    reranker = Reranker(settings.reranker_model, device=device, max_length=settings.rerank_max_length,
                        top_k=settings.rerank_top_k, gap_cutoff=settings.rerank_gap_cutoff).load()
    pipeline = RetrievalPipeline(hybrid, reranker)

    llm = LlamaCppClient(settings.llm_model_path, n_ctx=settings.llm_context_length,
                         n_gpu_layers=settings.llm_n_gpu_layers, max_tokens=settings.llm_max_tokens,
                         temperature=settings.llm_temperature, enable_thinking=settings.llm_enable_thinking).load()
    prompt_builder = PromptBuilder(
        ContextBuilder(llm.count_tokens, settings.context_max_tokens, settings.context_max_docs),
        history_turns=settings.history_turns,
    )
    print(f"all models loaded; VRAM used: {vram()} (aux device={device})")

    for q in queries:
        res = pipeline.search(q)
        messages, built = prompt_builder.build(q, res.docs)
        prompt_tokens = llm.count_tokens(llm.render(messages))
        r = llm.generate(messages)
        print("\n" + "=" * 78)
        print(f"Q: {q}")
        print(f"   retrieved: {[f'{d.chunk_id}({d.source})' for d in res.docs]}")
        print(f"   context: {len(built.included)} docs / {built.tokens} tok | prompt {prompt_tokens} tok | "
              f"gen {r.completion_tokens} tok in {r.latency_ms:.0f}ms ({r.completion_tokens/(r.latency_ms/1000):.0f} tok/s)")
        print("-" * 78)
        print(r.text)
    print(f"\nVRAM at end: {vram()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
