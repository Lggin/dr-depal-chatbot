"""Qdrant vector database client."""

import logging
from typing import List, Optional
import uuid
import numpy as np
from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.http import models

from app.rag.metadata_filter import INDEXED_KEYWORD_FIELDS, INDEXED_INTEGER_FIELDS, build_filter

logger = logging.getLogger(__name__)


class QdrantClient:
    """Qdrant vector database client."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "depalletizer_docs",
        embedding_dim: int = 1024,
    ):
        """
        Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            collection_name: Collection name for documents
            embedding_dim: Embedding dimension
        """
        self.url = url
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None

    def connect(self):
        """Connect to Qdrant server."""
        logger.info(f"Connecting to Qdrant at {self.url}")

        try:
            if self.url == ":memory:":
                self.client = QdrantClientBase(location=":memory:")  # for unit tests
            else:
                self.client = QdrantClientBase(url=self.url)

            collections = self.client.get_collections().collections
            logger.info(f"✅ Connected to Qdrant. Collections: {[c.name for c in collections]}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, force: bool = False):
        """
        Create collection for documents.

        Args:
            force: Delete existing collection if it exists
        """
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info(f"Creating collection: {self.collection_name}")

        # Check if collection exists
        try:
            existing = self.client.collection_exists(self.collection_name)
            if existing:
                if force:
                    logger.info("Deleting existing collection...")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return
        except Exception as e:
            logger.warning(f"Error checking collection: {e}")

        # Create collection
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            for field in INDEXED_KEYWORD_FIELDS:
                self.client.create_payload_index(
                    self.collection_name, field, models.PayloadSchemaType.KEYWORD
                )
            for field in INDEXED_INTEGER_FIELDS:
                self.client.create_payload_index(
                    self.collection_name, field, models.PayloadSchemaType.INTEGER
                )
            logger.info(f"✅ Collection created: {self.collection_name} (+{len(INDEXED_KEYWORD_FIELDS) + len(INDEXED_INTEGER_FIELDS)} payload indexes)")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def upsert_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[dict]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Upsert documents to collection.

        Args:
            texts: List of document texts
            embeddings: numpy array of embeddings
            metadata: List of metadata dicts
            chunk_ids: List of chunk IDs (generated if not provided)

        Returns:
            List of document IDs
        """
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        # Generate chunk IDs if not provided
        if chunk_ids is None:
            chunk_ids = [str(uuid.uuid4()) for _ in texts]

        # Prepare payload
        points = []
        for i, (text, embedding, chunk_id) in enumerate(zip(texts, embeddings, chunk_ids)):
            payload = {
                "text": text,
                "chunk_id": chunk_id,
            }

            # Add metadata if provided
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            # Deterministic UUID from chunk_id so re-ingesting the same chunk
            # overwrites rather than duplicates, and batches never collide.
            point = models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector=embedding.tolist(),
                payload=payload,
            )
            points.append(point)

        # Upsert to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(f"✅ Upserted {len(points)} documents")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Payload filter dict (see metadata_filter.build_filter)

        Returns:
            List of search results with score and payload
        """
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                query_filter=build_filter(filters),
                limit=top_k,
                with_payload=True,
            ).points

            search_results = []
            for result in results:
                search_results.append({
                    "score": result.score,
                    "chunk_id": result.payload.get("chunk_id"),
                    "text": result.payload.get("text"),
                    "metadata": {
                        k: v for k, v in result.payload.items()
                        if k not in ["text", "chunk_id"]
                    },
                })

            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def scroll_by_filter(self, filters: dict, limit: int = 20) -> List[dict]:
        """Exact-match lookup without a query vector (e.g. alarm_code == 'VAC-203')."""
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=build_filter(filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "chunk_id": p.payload.get("chunk_id"),
                "text": p.payload.get("text"),
                "metadata": {k: v for k, v in p.payload.items() if k not in ("text", "chunk_id")},
            }
            for p in points
        ]

    def scroll_all(self, page_size: int = 256) -> List[dict]:
        """Every point's chunk_id/text/metadata (no vectors) — feeds the BM25 index."""
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(
                {
                    "chunk_id": p.payload.get("chunk_id"),
                    "text": p.payload.get("text"),
                    "metadata": {k: v for k, v in p.payload.items() if k not in ("text", "chunk_id")},
                }
                for p in points
            )
            if offset is None:
                return out

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": str(info.status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise

    def delete_collection(self):
        """Delete collection."""
        if self.client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"✅ Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def close(self):
        """Close connection."""
        if self.client is not None:
            self.client.close()
            logger.info("Qdrant connection closed")
