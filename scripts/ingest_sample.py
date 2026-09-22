#!/usr/bin/env python3
"""Embed corpus/processed/sample_chunks.json and upsert into Qdrant.

Temporary until Phase 2's real ingestion pipeline; used by Steps 2-4 to test retrieval.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.dependencies import resolve_aux_device
from app.llm.embedding_client import EmbeddingClient
from app.rag.qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ingest_sample")

CHUNKS_PATH = Path(__file__).parent.parent / "corpus/processed/sample_chunks.json"


def main() -> int:
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    log.info(f"{len(chunks)} chunks from {CHUNKS_PATH.name}")

    embedder = EmbeddingClient(settings.embedding_model, device=resolve_aux_device())
    embedder.load()

    texts = [c["text"] for c in chunks]
    over = [(c["chunk_id"], embedder.count_tokens(c["text"])) for c in chunks
            if embedder.count_tokens(c["text"]) > embedder.max_seq_length]
    if over:
        log.warning(f"{len(over)} chunks exceed max_seq_length={embedder.max_seq_length}: {over}")
    vectors = embedder.embed(texts)

    store = QdrantClient(settings.qdrant_url, settings.qdrant_collection, embedder.embedding_dim)
    store.connect()
    store.create_collection(force=True)
    store.upsert_documents(
        texts=texts,
        embeddings=vectors,
        metadata=[{k: v for k, v in c.items() if k not in ("text", "chunk_id")} for c in chunks],
        chunk_ids=[c["chunk_id"] for c in chunks],
    )
    info = store.get_collection_info()
    log.info(f"collection {info['name']}: {info['points_count']} points, {info['vector_size']}d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
