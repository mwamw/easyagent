"""Layered-memory repositories; external clients are loaded only on use."""
from importlib import import_module

__all__ = ["DocumentStore", "SQLiteDocumentStore", "VectorStore", "QdrantVectorStore", "GraphStore", "Neo4jGraphStore"]

def __getattr__(name):
    if name in __all__:
        return getattr(import_module("." + name, __name__), name)
    raise AttributeError(name)
