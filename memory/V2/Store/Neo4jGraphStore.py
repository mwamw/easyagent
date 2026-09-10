"""Layered-memory entity repository using the public Neo4j property graph."""
from storage.neo4j import Neo4jGraphStore as Neo4jBackend
from memory.graph_repository import MemoryGraphRepository


class Neo4jGraphStore(MemoryGraphRepository):
    def __init__(self, uri, username, password, database="neo4j"):
        super().__init__(Neo4jBackend(uri, username, password, database=database, namespace="layered-memory"))

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
