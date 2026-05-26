"""
Embedding service using fastembed.
Converts text into vector embeddings for semantic search.
"""

from fastembed import TextEmbedding
from typing import List
from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings using fastembed."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = TextEmbedding(model_name=self.model_name)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        embeddings = list(self.model.embed([text]))
        if not embeddings:
            raise ValueError("Failed to generate embedding")
        return embeddings[0].tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Texts list cannot be empty")
        embeddings = list(self.model.embed(texts))
        return [embedding.tolist() for embedding in embeddings]


_embedding_service: EmbeddingService = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
