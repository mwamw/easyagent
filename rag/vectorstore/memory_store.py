"""RAG convenience index backed by the shared in-memory vector implementation."""
from storage.vector import InMemoryVectorStore
from rag.index import ChunkIndex


class MemoryVectorStore(ChunkIndex):
    def __init__(self):
        super().__init__(InMemoryVectorStore())
