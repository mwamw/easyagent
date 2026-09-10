"""Chroma vector backend with lossless JSON metadata and cosine scores."""

import json

from .vector import VectorStore
from .records import VectorRecord, VectorMatch, json_copy, validate_vector, validate_filter, validate_limit


class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name="easyagent", persist_directory=None, *, client=None):
        if client is None:
            import chromadb
            from chromadb.config import Settings
            settings = Settings(anonymized_telemetry=False)
            client = chromadb.PersistentClient(path=str(persist_directory), settings=settings) if persist_directory else chromadb.Client(settings)
        self._client = client
        self._collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _encode(metadata):
        metadata = json_copy(metadata)
        if any(key.startswith("_ea_") for key in metadata):
            raise ValueError("Metadata keys starting with _ea_ are reserved")
        values = {key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))}
        values["_ea_json"] = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        return values

    @staticmethod
    def _decode(metadata):
        values = metadata or {}
        return json.loads(values["_ea_json"]) if "_ea_json" in values else dict(values)

    @staticmethod
    def _where(where):
        validate_filter(where)
        clauses = []
        for key, value in (where or {}).items():
            if isinstance(value, dict):
                clauses.extend({key: {op: item}} for op, item in value.items())
            else:
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("Chroma filters require scalar values")
                clauses.append({key: {"$eq": value}})
        return {} if not clauses else {"where": clauses[0] if len(clauses) == 1 else {"$and": clauses}}

    def upsert(self, records):
        for record in records:
            validate_vector(record.vector)
        for start in range(0, len(records), 1000):
            batch = records[start:start + 1000]
            self._collection.upsert(ids=[r.id for r in batch], embeddings=[r.vector for r in batch],
                                    documents=[r.content for r in batch], metadatas=[self._encode(r.metadata) for r in batch])

    def search(self, vector, *, where=None, limit=5):
        validate_vector(vector)
        validate_limit(limit)
        filters = self._where(where)
        if not limit or not self.count():
            return []
        result = self._collection.query(query_embeddings=[vector], n_results=min(limit, self.count()), **filters)
        return [VectorMatch(record_id, 1.0 - float(result["distances"][0][i]),
                            self._decode(result["metadatas"][0][i]), (result["documents"][0][i] or ""))
                for i, record_id in enumerate(result["ids"][0])]

    def scan(self, *, where=None):
        filters = self._where(where)
        records = []
        offset = 0
        while True:
            batch = self._collection.get(limit=1000, offset=offset, include=["embeddings", "metadatas", "documents"], **filters)
            ids = batch["ids"]
            records.extend(VectorRecord(record_id, list(batch["embeddings"][i]), self._decode(batch["metadatas"][i]),
                                        batch["documents"][i] or "") for i, record_id in enumerate(ids))
            if len(ids) < 1000:
                return records
            offset += len(ids)

    def count(self):
        return self._collection.count()

    def delete(self, ids):
        if ids:
            self._collection.delete(ids=ids)
