#!/usr/bin/env python3
"""Step 3 check: sparse-only (BM25 + Kiwi) baseline on evaluation/sample_cases.py.

No embedding model needed — the BM25 index is built from the Qdrant payloads.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.rag.qdrant_client import QdrantClient
from app.rag.sparse_retriever import SparseRetriever
from evaluation.sample_cases import report

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim)
    store.connect()
    retriever = SparseRetriever.from_qdrant(store, top_k=12)
    print(f"BM25 index over {retriever.size} chunks")
    print("tokens('VAC203이 뭐야?') =", retriever.tokenizer.tokenize("VAC203이 뭐야?"))
    print("tokens('빨판이 박스를 못 잡아') =", retriever.tokenizer.tokenize("빨판이 박스를 못 잡아"))
    report("sparse", lambda q, f: retriever.retrieve(q, filters=f).chunk_ids())
    return 0


if __name__ == "__main__":
    sys.exit(main())
