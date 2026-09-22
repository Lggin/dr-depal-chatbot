"""Embedding model client."""

import logging
import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """BGE-M3 embedding model client."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        """
        Initialize embedding client.

        Args:
            model_name: Model identifier
            device: Device to load model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None

    def load(self):
        """Load embedding model."""
        logger.info(f"Loading embedding model: {self.model_name}")

        # Check GPU availability
        if self.device == "cuda":
            if not self._has_cuda():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"

        # Load model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            # Chunks longer than this are silently truncated at embed time —
            # the chunker must respect it (we once lost 21% of chunk tails this way).
            self.max_seq_length = self.model.max_seq_length
            logger.info(
                f"✅ Model loaded on {self.device}. dim={self.embedding_dim}, "
                f"max_seq_length={self.max_seq_length} tokens"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not texts:
            return np.array([])

        logger.debug(f"Embedding {len(texts)} texts with batch_size={batch_size}")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed single text.

        Args:
            text: Text to embed

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def count_tokens(self, text: str) -> int:
        """Token count under this model's tokenizer (for chunk-size decisions)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
