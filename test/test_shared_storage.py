"""Deterministic storage/RAG contracts: no downloaded models or paid calls."""

from datetime import datetime
import pytest

from easyagent.storage import DocumentRecord, InMemoryDocumentStore, SQLiteDocumentStore, InMemoryVectorStore, VectorRecord, EmbeddingModel
from memory.repositories import MemoryDocumentRepository
from rag.index import ChunkIndex
from rag.document import Document_Chunk
from rag.retriever.vector_retriever import VectorRetriever


@pytest.fixture(params=["memory", "sqlite"])
def documents(request, tmp_path):
    return InMemoryDocumentStore() if request.param == "memory" else SQLiteDocumentStore(tmp_path / "documents.db")


def test_documents_round_trip_and_filter(documents):
    record = DocumentRecord("one", "content", {"user": "a", "n": 3, "nested": {"items": [1, None]}})
    documents.put(record)
    assert documents.get("one") == record
    record.metadata["user"] = "changed"
    assert documents.get("one").metadata["user"] == "a"
    assert len(documents.query(where={"user": "a", "n": {"$gte": 2, "$lte": 4}})) == 1
    assert documents.query(where={"user": "b"}) == []
    with pytest.raises(ValueError):
        documents.query(where={"user": {"$unsupported": "a"}})
    assert documents.delete("one")
    assert documents.get("one") is None


@pytest.mark.asyncio
async def test_sqlite_async_bridge_does_not_share_thread_affine_connections(documents):
    await documents.aput(DocumentRecord("async", "value"))
    assert (await documents.aget("async")).content == "value"
    assert await documents.adelete("async")


def test_sqlite_namespace_isolation(tmp_path):
    first = SQLiteDocumentStore(tmp_path / "shared.db", namespace="a")
    second = SQLiteDocumentStore(tmp_path / "shared.db", namespace="b")
    first.put(DocumentRecord("id", "first"))
    second.put(DocumentRecord("id", "second"))
    first.delete("id")
    assert second.get("id").content == "second"


def test_memory_mapping_lives_outside_storage(documents):
    repository = MemoryDocumentRepository(documents)
    repository.add_memory("mem", "alice", "facts", "episodic", datetime.now(), .7, {"session_id": "s", "extra": [1, 2]})
    item = repository.get_memory("mem")
    assert item.content == "facts"
    assert item.metadata == {"session_id": "s", "extra": [1, 2]}
    assert repository.search_memory(session_id="other") == []
    assert len(repository.search_memory(user_id="alice", session_id="s", importance_threshold=.6)) == 1
    assert not hasattr(documents, "add_memory")


def test_generic_vectors_are_independent_of_rag_and_memory():
    store = InMemoryVectorStore()
    store.upsert([VectorRecord("one", [1., 0.], {"scope": "a"}, "one"), VectorRecord("two", [0., 1.], {"scope": "b"}, "two")])
    assert store.search([1., 0.])[0].id == "one"
    assert store.search([1., 0.], where={"scope": "b"})[0].id == "two"
    assert store.search([1., 0.], limit=0) == []
    with pytest.raises(ValueError):
        store.upsert([VectorRecord("bad", [1., 2., 3.])])
    assert store.count() == 2
    store.clear(where={"scope": "a"})
    assert [record.id for record in store.scan()] == ["two"]


class StubEmbedding(EmbeddingModel):
    dimension = 2

    def embed_documents(self, texts):
        return [[1., 0.] if "alpha" in text else [0., 1.] for text in texts]


def test_rag_uses_shared_backend_and_preserves_domain_metadata():
    store = InMemoryVectorStore()
    index = ChunkIndex(store)
    chunk = Document_Chunk(document_id="doc", document_path="doc.md", chunk_id="c", chunk_index=0, content="alpha", metadata={"nested": [1, {"x": True}]})
    index.add_documents([chunk], [[1., 0.]])
    retrieved = VectorRetriever(store, StubEmbedding()).retrieve("alpha")[0]
    assert retrieved.content == chunk.content
    assert retrieved.metadata == chunk.metadata
    assert retrieved.chunk_id == "c"


def test_local_qdrant_contract():
    pytest.importorskip("qdrant_client")
    from easyagent.storage import QdrantVectorStore
    store = QdrantVectorStore("test_contract", dimension=2)
    try:
        store.upsert([VectorRecord("arbitrary-document-id", [1., 0.], {"nested": [1, 2], "owner": "a"})])
        match = store.search([1., 0.], where={"owner": "a"})[0]
        assert match.id == "arbitrary-document-id"
        assert match.metadata["nested"] == [1, 2]
        assert store.search([1., 0.], where={"owner": "b"}) == []
        store.delete([match.id])
        assert store.count() == 0
    finally:
        store.close()


def test_local_chroma_round_trip_without_embedding_model(tmp_path):
    pytest.importorskip("chromadb")
    from easyagent.storage import ChromaVectorStore
    store = ChromaVectorStore("contract_store", str(tmp_path / "chroma"))
    store.upsert([VectorRecord("id", [1., 0.], {"owner": "a", "nested": [{"ok": True}]}, "content")])
    match = store.search([1., 0.], where={"owner": "a"})[0]
    assert match.metadata["nested"] == [{"ok": True}]
    assert match.content == "content"
    assert store.scan()[0].id == "id"
    store.delete(["id"])
    assert store.count() == 0


def test_legacy_sqlite_memory_migration_preserves_source_and_properties(tmp_path):
    import json
    import sqlite3
    from easyagent.memory import migrate_memory_sqlite
    source, target = tmp_path / "old.db", tmp_path / "new.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE memories(id TEXT,user_id TEXT,content TEXT,memory_type TEXT,timestamp INTEGER,importance REAL,properties TEXT)")
        conn.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?)", ("id", "u", "fact", "episodic", 100, .7, json.dumps({"session_id": "s", "extra": [1]})))
    original = source.read_bytes()
    assert migrate_memory_sqlite(source, target)["count"] == 1
    assert not target.exists()
    migrate_memory_sqlite(source, target, dry_run=False)
    assert source.read_bytes() == original
    repository = MemoryDocumentRepository(SQLiteDocumentStore(target, namespace="layered-memory"))
    assert repository.get_memory("id").metadata == {"session_id": "s", "extra": [1]}


def test_legacy_graph_migration_only_reads_source():
    from easyagent.memory import migrate_legacy_graph
    class Source:
        def __init__(self):
            self.queries = []
        def session(self, **kwargs):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def run(self, query):
            self.queries.append(query)
            if "properties(n)" in query:
                return [{"id": "a", "properties": {"name": "A"}}, {"id": "b", "properties": {"name": "B"}}]
            return [{"source": "a", "target": "b", "kind": "related", "properties": {"evidence": "old"}}]
    class Target:
        def __init__(self):
            self.saved_nodes, self.saved_edges = [], []
        def nodes(self):
            return self.saved_nodes
        def edges(self):
            return self.saved_edges
        def put_node(self, node):
            self.saved_nodes.append(node)
        def put_edge(self, edge):
            self.saved_edges.append(edge)
    source, target = Source(), Target()
    assert migrate_legacy_graph(source, target)["nodes"] == 2
    assert target.nodes() == []
    migrate_legacy_graph(source, target, dry_run=False)
    assert target.edges()[0].properties == {"evidence": "old"}
    assert all(query.startswith("MATCH") for query in source.queries)
    with pytest.raises(ValueError, match="empty target"):
        migrate_legacy_graph(source, target, dry_run=False)
