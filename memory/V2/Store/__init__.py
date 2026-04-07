try:
    # 尝试相对导入
    from .DocumentStore import DocumentStore
    from .SQLiteDocumentStore import SQLiteDocumentStore
    from .VectorStore import VectorStore
    from .QdrantVectorStore import QdrantVectorStore
except ImportError:
    # When memory/V2 is a sys.path entry (bare imports)
    from Store.DocumentStore import DocumentStore
    from Store.SQLiteDocumentStore import SQLiteDocumentStore
    from Store.VectorStore import VectorStore
    from Store.QdrantVectorStore import QdrantVectorStore

try:
    try:
        from .Neo4jGraphStore import Neo4jGraphStore
    except ImportError:
        from Store.Neo4jGraphStore import Neo4jGraphStore
except ImportError:
    Neo4jGraphStore = None  # type: ignore