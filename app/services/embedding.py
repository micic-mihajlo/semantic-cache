"""Embedding service using SentenceTransformer for semantic similarity."""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Singleton wrapper for SentenceTransformer model."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        """Generate normalized 384-dim embedding for text."""
        return self.model.encode(text, normalize_embeddings=True)


# Global singleton instance
embedding_service = EmbeddingService()
