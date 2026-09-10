"""RAG-domain conversion over a shared, domain-neutral VectorStore."""

from storage import VectorRecord
from .document import Document_Chunk


class ChunkIndex:
    def __init__(self, store):
        self.store = store

    def add_documents(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        records = [VectorRecord(chunk.chunk_id, list(vector), {
            **chunk.metadata,
            "document_id": chunk.document_id, "document_path": chunk.document_path,
            "chunk_index": chunk.chunk_index, "chunk_metadata": chunk.metadata,
        }, chunk.content) for chunk, vector in zip(chunks, embeddings)]
        self.store.upsert(records)

    @staticmethod
    def _chunk(match):
        meta = match.metadata
        return Document_Chunk(document_id=meta["document_id"], document_path=meta["document_path"],
                              chunk_id=match.id, chunk_index=meta["chunk_index"], content=match.content,
                              metadata=dict(meta.get("chunk_metadata", meta)))

    def similarity_search_with_score(self, vector, k=4, filter=None):
        return [(self._chunk(match), match.score) for match in self.store.search(vector, where=filter, limit=k)]

    def similarity_search(self, vector, k=4, filter=None):
        return [chunk for chunk, _ in self.similarity_search_with_score(vector, k, filter)]

    def delete(self, ids):
        self.store.delete(ids)

    def clear(self):
        self.store.clear()

    def count(self):
        return self.store.count()
