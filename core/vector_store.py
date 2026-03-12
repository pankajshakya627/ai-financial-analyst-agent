"""
ChromaDB vector store manager for the RAG pipeline.
Handles document storage, retrieval, and collection management.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from core.embeddings import EmbeddingEngine, get_embedding_engine

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store."""

    content: str
    metadata: dict
    score: float  # similarity score (higher = more relevant)
    doc_id: str


@dataclass
class DocumentChunk:
    """A chunk of text ready for insertion into the vector store."""

    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: Optional[str] = None


class ChromaEmbeddingFunction:
    """Wraps our EmbeddingEngine for ChromaDB compatibility."""

    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.engine.embed_texts(input)

    def name(self) -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "ollama"


class VectorStoreManager:
    """Manages ChromaDB collections for financial document retrieval."""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.chroma_ef = ChromaEmbeddingFunction(self.embedding_engine)

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB initialized at: {settings.chroma_persist_dir}")

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a named collection."""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        collection_name: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """Add document chunks to a collection. Returns count of added docs."""
        collection = self.get_or_create_collection(collection_name)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = chunk.doc_id or f"{collection_name}_{collection.count() + i}"
            ids.append(doc_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)

        if not documents:
            return 0

        # ChromaDB handles batching internally
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Added {len(documents)} chunks to collection '{collection_name}'")
        return len(documents)

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: Optional[int] = None,
        where_filter: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """Query a collection and return ranked results."""
        collection = self.get_or_create_collection(collection_name)
        top_k = top_k or settings.top_k_results

        if collection.count() == 0:
            logger.warning(f"Collection '{collection_name}' is empty")
            return []

        query_params = {
            "query_texts": [query_text],
            "n_results": min(top_k, collection.count()),
        }
        if where_filter:
            query_params["where"] = where_filter

        results = collection.query(**query_params)

        retrieved = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results["distances"] else 1.0
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1.0 - (distance / 2.0)

            if score >= settings.similarity_threshold:
                retrieved.append(
                    RetrievedDocument(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=score,
                        doc_id=results["ids"][0][i],
                    )
                )

        retrieved.sort(key=lambda x: x.score, reverse=True)
        return retrieved

    def multi_collection_query(
        self,
        collection_names: list[str],
        query_text: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedDocument]:
        """Query across multiple collections and merge results."""
        all_results = []
        for name in collection_names:
            results = self.query(name, query_text, top_k=top_k)
            all_results.extend(results)

        # Re-rank by score across collections
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[: (top_k or settings.top_k_results)]

    def delete_collection(self, name: str) -> None:
        """Delete an entire collection."""
        self.client.delete_collection(name)
        logger.info(f"Deleted collection: {name}")

    def list_collections(self) -> list[str]:
        """List all collection names."""
        return [c.name for c in self.client.list_collections()]

    def get_collection_stats(self, name: str) -> dict:
        """Get statistics about a collection."""
        collection = self.get_or_create_collection(name)
        return {
            "name": name,
            "count": collection.count(),
            "metadata": collection.metadata,
        }
