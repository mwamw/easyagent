"""Document persistence independent of memory and retrieval domain models."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import asdict
from pathlib import Path
import sqlite3
import json
from threading import RLock
from contextlib import contextmanager, closing

from .records import DocumentRecord, json_copy, matches, validate_filter, validate_limit


class DocumentStore(ABC):
    @abstractmethod
    def put(self, record: DocumentRecord) -> None: ...

    @abstractmethod
    def get(self, record_id: str) -> DocumentRecord | None: ...

    @abstractmethod
    def query(self, *, where=None, limit=1000) -> list[DocumentRecord]: ...

    @abstractmethod
    def delete(self, record_id: str) -> bool: ...

    async def aput(self, record):
        return await asyncio.to_thread(self.put, record)

    async def aget(self, record_id):
        return await asyncio.to_thread(self.get, record_id)

    async def aquery(self, **kwargs):
        return await asyncio.to_thread(self.query, **kwargs)

    async def adelete(self, record_id):
        return await asyncio.to_thread(self.delete, record_id)

    def close(self):
        pass

    async def aclose(self):
        await asyncio.to_thread(self.close)


class InMemoryDocumentStore(DocumentStore):
    def __init__(self):
        self._records = {}
        self._lock = RLock()

    def put(self, record):
        data = json_copy(asdict(record))
        if not record.id:
            raise ValueError("Record ID must not be empty")
        with self._lock:
            self._records[record.id] = data

    def get(self, record_id):
        with self._lock:
            data = self._records.get(record_id)
            return DocumentRecord(**json_copy(data)) if data is not None else None

    def query(self, *, where=None, limit=1000):
        validate_filter(where)
        validate_limit(limit)
        with self._lock:
            return [DocumentRecord(**json_copy(data)) for data in self._records.values()
                    if matches(data["metadata"], where)][:limit]

    def delete(self, record_id):
        with self._lock:
            return self._records.pop(record_id, None) is not None


class SQLiteDocumentStore(DocumentStore):
    """A connection per operation permits the async thread bridge safely.

    A distinct documents table leaves legacy memories tables untouched.
    """

    def __init__(self, db_path, *, namespace="default"):
        self.db_path = str(db_path)
        if self.db_path == ":memory:":
            raise ValueError("Use InMemoryDocumentStore for non-persistent storage")
        self.namespace = namespace
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS documents (namespace TEXT NOT NULL, id TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, PRIMARY KEY(namespace,id))")

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            with conn:
                yield conn

    def put(self, record):
        if not record.id:
            raise ValueError("Record ID must not be empty")
        data = json.dumps(record.metadata, ensure_ascii=False, allow_nan=False)
        with self._connect() as conn:
            conn.execute("INSERT INTO documents VALUES (?,?,?,?) ON CONFLICT(namespace,id) DO UPDATE SET content=excluded.content, metadata=excluded.metadata",
                         (self.namespace, record.id, record.content, data))

    def get(self, record_id):
        with self._connect() as conn:
            row = conn.execute("SELECT id,content,metadata FROM documents WHERE namespace=? AND id=?", (self.namespace, record_id)).fetchone()
        return DocumentRecord(row[0], row[1], json.loads(row[2])) if row else None

    def query(self, *, where=None, limit=1000):
        validate_filter(where)
        validate_limit(limit)
        result = []
        if not limit:
            return result
        with self._connect() as conn:
            for record_id, content, metadata in conn.execute("SELECT id,content,metadata FROM documents WHERE namespace=? ORDER BY rowid", (self.namespace,)):
                decoded = json.loads(metadata)
                if matches(decoded, where):
                    result.append(DocumentRecord(record_id, content, decoded))
                    if len(result) >= limit:
                        break
        return result

    def delete(self, record_id):
        with self._connect() as conn:
            return conn.execute("DELETE FROM documents WHERE namespace=? AND id=?", (self.namespace, record_id)).rowcount > 0
