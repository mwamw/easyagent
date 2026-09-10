"""Layered-memory domain mapping. Storage itself knows nothing about memory."""

from datetime import datetime

from storage import DocumentRecord, DocumentStore, VectorRecord, VectorStore
from .V2.BaseMemory import MemoryItem


def _json_metadata(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    return value


class MemoryDocumentRepository:
    def __init__(self, store: DocumentStore):
        self.store = store

    @staticmethod
    def _memory(record):
        if record is None:
            return None
        meta = record.metadata
        return MemoryItem(id=record.id, content=record.content, type=meta["memory_type"],
                          user_id=meta["user_id"], timestamp=datetime.fromtimestamp(meta["timestamp"]),
                          importance=meta["importance"], metadata=meta.get("properties") or {})

    def add_memory(self, memory_id, user_id, content, memory_type, timestamp, importance, properties=None):
        properties = _json_metadata(properties or {})
        self.store.put(DocumentRecord(memory_id, content, {
            "user_id": user_id, "memory_type": str(getattr(memory_type, "value", memory_type)),
            "timestamp": timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp,
            "importance": importance, "session_id": properties.get("session_id"), "properties": properties,
        }))
        return memory_id

    def get_memory(self, memory_id):
        return self._memory(self.store.get(memory_id))

    def remove_memory(self, memory_id):
        return self.store.delete(memory_id)

    def update_memory(self, memory_id, content=None, importance=None, properties=None):
        record = self.store.get(memory_id)
        if record is None:
            return False
        if content is not None:
            record.content = content
        if importance is not None:
            record.metadata["importance"] = importance
        if properties is not None:
            record.metadata["properties"] = _json_metadata(properties)
            record.metadata["session_id"] = properties.get("session_id")
        self.store.put(record)
        return True

    def search_memory(self, user_id=None, memory_type=None, session_id=None, start_ts=None, end_ts=None, importance_threshold=None, limit=1000):
        where = {}
        for key, value in (("user_id", user_id), ("memory_type", memory_type), ("session_id", session_id)):
            if value is not None:
                where[key] = getattr(value, "value", value)
        if start_ts is not None or end_ts is not None:
            where["timestamp"] = {}
            if start_ts is not None:
                where["timestamp"]["$gte"] = start_ts
            if end_ts is not None:
                where["timestamp"]["$lte"] = end_ts
        if importance_threshold is not None:
            where["importance"] = {"$gte": importance_threshold}
        return [self._memory(record) for record in self.store.query(where=where, limit=limit)]

    def get_all_memories(self):
        return [self._memory(record) for record in self.store.query(limit=2**31 - 1)]

    def clear_type_memory(self, memory_type):
        for record in self.store.query(where={"memory_type": getattr(memory_type, "value", memory_type)}, limit=2**31 - 1):
            self.store.delete(record.id)

    def clear_all(self):
        for record in self.store.query(limit=2**31 - 1):
            self.store.delete(record.id)

    def get_database_stats(self):
        return {"count": len(self.store.query(limit=2**31 - 1))}


class MemoryVectorIndex:
    def __init__(self, store: VectorStore):
        self.store = store

    def add_vectors(self, vectors, metadata, ids):
        if len(vectors) != len(metadata) or len(ids) != len(vectors):
            raise ValueError("Vector IDs, vectors, and metadata must have equal lengths")
        self.store.upsert([VectorRecord(record_id, list(vector), _json_metadata(meta)) for record_id, vector, meta in zip(ids, vectors, metadata)])

    def remove_vectors(self, ids):
        self.store.delete(ids)
        return len(ids)

    def clear_type_memory(self, memory_type):
        self.store.clear(where={"memory_type": getattr(memory_type, "value", memory_type)})

    def get_collection_stats(self):
        return {"count": self.store.count()}

    def search_similar(self, query_embedding, where=None, limit=5):
        return [{"memory_id": match.id, "score": match.score, "metadata": match.metadata}
                for match in self.store.search(query_embedding, where=where, limit=limit)]

    def get_all_vectors(self, with_vector=False):
        return [{"memory_id": record.id, "metadata": record.metadata,
                 **({"vector": record.vector} if with_vector else {})} for record in self.store.scan()]


def document_repository(store):
    return MemoryDocumentRepository(store) if isinstance(store, DocumentStore) else store


def vector_index(store):
    return MemoryVectorIndex(store) if isinstance(store, VectorStore) else store
