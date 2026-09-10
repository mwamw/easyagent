# 公共存储与 Embedding

公共入口是 easyagent.storage；实现位于 storage/，可独立于 Agent 使用。

```python
from easyagent.storage import SQLiteDocumentStore, DocumentRecord, InMemoryVectorStore, VectorRecord

documents = SQLiteDocumentStore("./records.db", namespace="notes")
documents.put(DocumentRecord("n1", "Example", {"owner": "alice", "tags": ["demo"]}))
assert documents.query(where={"owner": "alice"})[0].id == "n1"

vectors = InMemoryVectorStore()
vectors.upsert([VectorRecord("n1", [1.0, 0.0], {"owner": "alice"})])
assert vectors.search([1.0, 0.0], where={"owner": "alice"})[0].id == "n1"
```

- 文档：DocumentStore、InMemoryDocumentStore、SQLiteDocumentStore。
- 向量：VectorStore、InMemoryVectorStore、QdrantVectorStore、ChromaVectorStore。
- 图：GraphStore、GraphNode、GraphEdge、Neo4jGraphStore。
- 嵌入：EmbeddingModel、OpenAIEmbedding、HuggingFaceEmbedding。

metadata 保存 JSON 数据，不能静默把任意对象字符串化。
文档/向量过滤支持等值、$in、$gte、$lte；不支持的条件报错。
向量后端使用 cosine，高分在前；维度与非有限值会校验。
本轮 Neo4j 后端使用 namespaced 固定标签，领域属性放在 JSON payload；
不是任意 Cypher 查询或跨后端分布式事务系统。

可选客户端只在实例化对应后端时导入。不要在基础模块导入时下载模型。
Embedding 区分 embed_documents 与 embed_query，异步入口加 a 前缀。
未知 OpenAI 模型的 dimension 在首次显式嵌入前报错，不偷偷发请求探测。

RAGPipeline/VectorRetriever 可接受公共 VectorStore；ChunkIndex 负责 Document_Chunk 转换。
rag.vectorstore 下的类是 RAG 便捷索引，不再持有另一套底层存储实现。
Memory 内部 repository 同样只负责领域转换。

## 旧数据

旧 SQLite memories 表不由公共后端自动读取；使用显式迁移：
python -m easyagent.migrate memory-sqlite --source old.db --target new.db --apply。
省略 --apply 只预览；源数据保留。

Qdrant 的旧 UUID/metadata 记录和 Chroma 的旧文档 metadata 可以读取；
若旧 Chroma metadata 已被字符串化，无法自动还原原始类型。
旧 Neo4j 图不与新 namespace/标签混写。使用 easyagent.memory.migrate_legacy_graph，
传入源 driver 和新 namespace 的 GraphStore；默认 dry_run=True。
它保留源图，显式应用时拒绝非空目标。跨服务器写入不承诺事务回滚；
验证节点、边及属性后再切换配置。过滤不是鉴权，应用仍需控制后端访问权限。
