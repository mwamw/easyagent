"""RAG document index backed by the public Chroma backend."""
from storage.chroma import ChromaVectorStore as ChromaBackend
from rag.index import ChunkIndex


class ChromaVectorStore(ChunkIndex):
    def __init__(self, collection_name="easyagent_rag", persist_directory=None):
        super().__init__(ChromaBackend(collection_name, persist_directory))
