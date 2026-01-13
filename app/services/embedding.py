"""Embedding service using SentenceTransformer for semantic similarity."""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrapper for SentenceTransformer model."""

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        """Generate normalized 384-dim embedding for text."""
        return self.model.encode(text, normalize_embeddings=True)


# Module-level singleton
embedding_service = EmbeddingService()
