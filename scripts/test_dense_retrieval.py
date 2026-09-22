#!/usr/bin/env python3
"""Step 2 check: dense-only baseline on evaluation/sample_cases.py."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.dependencies import resolve_aux_device
from app.llm.embedding_client import EmbeddingClient
from app.rag.dense_retriever import DenseRetriever
from app.rag.qdrant_client import QdrantClient
from evaluation.sample_cases import report

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    embedder = EmbeddingClient(settings.embedding_model, device=resolve_aux_device())
    embedder.load()
    store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, embedder.embedding_dim)
    store.connect()
    retriever = DenseRetriever(embedder, store, top_k=12)
    report("dense", lambda q, f: retriever.retrieve(q, filters=f).chunk_ids())
    return 0


if __name__ == "__main__":
    sys.exit(main())
