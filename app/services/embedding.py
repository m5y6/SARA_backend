"""
Embedding service using fastembed.
Converts text into vector embeddings for semantic search.
"""

import os
from typing import List, Any
from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings using fastembed."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        # Import fastembed lazily so the app can start even if the package
        # is not installed. If it's missing, we keep `self.model` as None
        # and raise a clear error when embedding is attempted.
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ModuleNotFoundError:
            self.model = None
            return

        # Define a consistent cache directory within the project.
        # fastembed will check this directory for a pre-existing model
        # and download it here if it's missing.
        model_cache_dir = os.path.abspath("app/models")
        print(f"Fastembed cache directory is set to: {model_cache_dir}")

        # Always try to load the model by name. fastembed will handle caching.
        self.model: Any = TextEmbedding(model_name=self.model_name, cache_dir=model_cache_dir)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if self.model is None:
            raise RuntimeError("fastembed is not installed; install it to use embeddings")

        embeddings = list(self.model.embed([text]))
        if not embeddings:
            raise ValueError("Failed to generate embedding")
        return embeddings[0].tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if self.model is None:
            raise RuntimeError("fastembed is not installed; install it to use embeddings")

        embeddings = list(self.model.embed(texts))
        return [embedding.tolist() for embedding in embeddings]


_embedding_service: EmbeddingService = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
