"""Layered-memory repository over the common SQLite document backend."""
from storage.document import SQLiteDocumentStore as SQLiteBackend
from memory.repositories import MemoryDocumentRepository


class SQLiteDocumentStore(MemoryDocumentRepository):
    def __init__(self, db_path):
        super().__init__(SQLiteBackend(db_path, namespace="layered-memory"))
