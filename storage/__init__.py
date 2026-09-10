"""Publicly reusable storage infrastructure. Optional clients are loaded lazily."""

from importlib import import_module
from .records import DocumentRecord, VectorRecord, VectorMatch, StorageError
from .document import DocumentStore, InMemoryDocumentStore, SQLiteDocumentStore
from .vector import VectorStore, InMemoryVectorStore
from .embedding import EmbeddingModel

_BACKENDS = {"OpenAIEmbedding": "openai_embedding", "HuggingFaceEmbedding": "huggingface_embedding",
             "ChromaVectorStore": "chroma", "QdrantVectorStore": "qdrant",
             "GraphStore": "graph", "GraphNode": "graph", "GraphEdge": "graph",
             "Neo4jGraphStore": "neo4j"}

__all__ = ["DocumentRecord", "VectorRecord", "VectorMatch", "StorageError", "DocumentStore",
           "InMemoryDocumentStore", "SQLiteDocumentStore", "VectorStore", "InMemoryVectorStore",
           "EmbeddingModel", *_BACKENDS]


def __getattr__(name):
    if name in _BACKENDS:
        return getattr(import_module("." + _BACKENDS[name], __name__), name)
    raise AttributeError(name)
