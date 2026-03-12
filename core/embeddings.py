"""
Embedding engine for the RAG pipeline.
Supports sentence-transformers (local), OpenAI, and Ollama embeddings.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config.settings import settings, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingEngine(ABC):
    """Abstract embedding engine."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class SentenceTransformerEmbedding(EmbeddingEngine):
    """Local sentence-transformers embedding (no API calls)."""

    def __init__(self, model_name: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {self.model_name} (dim={self._dimension})")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedding(EmbeddingEngine):
    """OpenAI text-embedding API."""

    def __init__(self, model_name: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = model_name or "text-embedding-3-small"
        self._dimension = 1536 if "small" in self.model_name else 3072
        logger.info(f"Initialized OpenAI embeddings: {self.model_name}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # OpenAI API supports batch embedding
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=[query])
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbedding(EmbeddingEngine):
    """
    Ollama-hosted embedding model.
    Uses models like nomic-embed-text, mxbai-embed-large, etc.
    Connects to a running Ollama server.
    """

    def __init__(self, model_name: Optional[str] = None):
        import httpx

        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model_name = model_name or settings.ollama_embedding_model
        self._dimension = None

        # Probe dimension by embedding a test string
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": ["test"]},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [[]])
            self._dimension = len(embeddings[0]) if embeddings and embeddings[0] else 768
        except Exception as e:
            logger.warning(f"Could not probe Ollama embedding dimension: {e}. Defaulting to 768.")
            self._dimension = 768

        logger.info(f"Initialized Ollama embeddings: {self.model_name} (dim={self._dimension})")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import httpx

        # Ollama /api/embed supports batch input
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])

    def embed_query(self, query: str) -> list[float]:
        result = self.embed_texts([query])
        return result[0] if result else [0.0] * self._dimension

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_engine(provider: Optional[EmbeddingProvider] = None) -> EmbeddingEngine:
    """Factory function to get the configured embedding engine."""
    provider = provider or settings.embedding_provider
    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedding()
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedding()
    elif provider == EmbeddingProvider.OLLAMA:
        return OllamaEmbedding()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
