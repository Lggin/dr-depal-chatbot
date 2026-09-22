#!/usr/bin/env python3
"""Test embedding and Qdrant indexing."""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm.embedding_client import EmbeddingClient
from app.rag.qdrant_client import QdrantClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test embedding and indexing."""
    logger.info("=" * 60)
    logger.info("Step 1: Embedding & Indexing Test")
    logger.info("=" * 60)

    # Sample documents
    sample_docs = [
        "VAC-203: 진공압이 정상 범위에 도달하지 못함",
        "Vacuum Pressure Low alarm indicates suction system failure",
        "박스를 흡착하지 못할 때는 진공 시스템을 확인하세요",
        "로봇 티칭은 teach pendant를 사용합니다",
        "비전 시스템이 박스를 정상적으로 감지하고 있는지 확인하세요",
    ]

    try:
        # Step 1a: Load embedding model
        logger.info("\n[1a] Loading embedding model...")
        embed_client = EmbeddingClient(
            model_name="BAAI/bge-m3",
            device="cuda",
        )
        embed_client.load()

        # Step 1b: Generate embeddings
        logger.info("\n[1b] Generating embeddings...")
        embeddings = embed_client.embed(sample_docs)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"   Shape: {embeddings.shape}")

        # Step 1c: Connect to Qdrant
        logger.info("\n[1c] Connecting to Qdrant...")
        qdrant = QdrantClient(
            url="http://localhost:6333",
            collection_name="depalletizer_docs",
            embedding_dim=1024,
        )
        qdrant.connect()

        # Step 1d: Create collection
        logger.info("\n[1d] Creating collection...")
        qdrant.create_collection(force=True)

        # Step 1e: Upsert documents
        logger.info("\n[1e] Upserting documents...")
        chunk_ids = qdrant.upsert_documents(
            texts=sample_docs,
            embeddings=embeddings,
            metadata=[
                {"alarm_code": "VAC-203", "language": "ko"},
                {"language": "en"},
                {"language": "ko"},
                {"language": "ko"},
                {"language": "ko"},
            ],
            chunk_ids=[f"doc_{i:03d}" for i in range(len(sample_docs))],
        )

        # Step 1f: Get collection info
        logger.info("\n[1f] Collection info...")
        info = qdrant.get_collection_info()
        logger.info(f"   Points: {info['points_count']} | status: {info['status']}")
        logger.info(f"   Vector: {info['vector_size']}d {info['distance']}")

        # Step 1g: Test search
        logger.info("\n[1g] Testing search...")
        query_text = "진공 시스템 문제"
        query_embedding = embed_client.embed_single(query_text)
        results = qdrant.search(query_embedding, top_k=3)

        logger.info(f"   Query: '{query_text}'")
        logger.info(f"   Results:")
        for i, result in enumerate(results, 1):
            logger.info(f"   {i}. Score: {result['score']:.4f}")
            logger.info(f"      ID: {result['chunk_id']}")
            logger.info(f"      Text: {result['text'][:50]}...")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Step 1 Test Completed Successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
