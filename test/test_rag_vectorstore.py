"""
RAG 向量存储单元测试
"""
import unittest
import sys
import os
import uuid
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag.document import Document_Chunk
from rag.vectorstore.memory_store import MemoryVectorStore
from rag.embedding.huggingface_embedding import HuggingFaceEmbedding


# 全局共享嵌入模型，避免重复加载
_embedding = None

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbedding()
    return _embedding


def make_chunk(content: str, index: int = 0, doc_id: str = "doc-001", **meta) -> Document_Chunk:
    """创建测试用文档块"""
    metadata = {"source": "/test/doc.txt"}
    metadata.update(meta)
    return Document_Chunk(
        document_id=doc_id,
        document_path="/test/doc.txt",
        chunk_id=str(uuid.uuid4()),
        content=content,
        metadata=metadata,
        chunk_index=index,
    )


def embed_chunks(chunks):
    """使用真实嵌入模型为文档块生成嵌入向量"""
    emb = get_embedding()
    texts = [c.content for c in chunks]
    return emb.embed_documents(texts)


# ============================================================
# MemoryVectorStore 测试
# ============================================================
class TestMemoryVectorStore(unittest.TestCase):
    """MemoryVectorStore 内存向量存储测试"""

    def setUp(self):
        self.store = MemoryVectorStore()
        self.emb = get_embedding()

    def test_add_and_count(self):
        """测试添加文档块并计数"""
        chunks = [make_chunk(f"这是第{i}段不同的测试内容", i) for i in range(5)]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)
        self.assertEqual(self.store.count(), 5)

    def test_add_multiple_batches(self):
        """测试多次添加"""
        chunks1 = [make_chunk("第一批次的文档内容", 0)]
        chunks2 = [make_chunk("第二批次的文档内容", 1)]
        self.store.add_documents(chunks1, embed_chunks(chunks1))
        self.store.add_documents(chunks2, embed_chunks(chunks2))
        self.assertEqual(self.store.count(), 2)

    def test_similarity_search(self):
        """测试相似度搜索"""
        chunks = [
            make_chunk("人工智能是计算机科学的分支", 0),
            make_chunk("机器学习使用统计方法", 1),
            make_chunk("深度学习是神经网络的应用", 2),
            make_chunk("今天天气真好适合出门", 3),
            make_chunk("自然语言处理是AI的应用", 4),
        ]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)

        query_emb = self.emb.embed_query("人工智能技术")
        results = self.store.similarity_search(query_emb, k=3)
        self.assertEqual(len(results), 3)

    def test_similarity_search_with_score(self):
        """测试带分数的相似度搜索"""
        chunks = [
            make_chunk("机器学习是AI的子领域", 0),
            make_chunk("烹饪美食是一种艺术", 1),
        ]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)

        # 使用第一个块的嵌入来搜索，自身应得最高分
        results = self.store.similarity_search_with_score(embeddings[0], k=2)
        self.assertEqual(len(results), 2)
        for chunk, score in results:
            self.assertIsInstance(chunk, Document_Chunk)
            self.assertIsInstance(score, float)
        # 自身的相似度应该接近 1.0
        self.assertAlmostEqual(results[0][1], 1.0, places=3)

    def test_similarity_search_sorted_by_score(self):
        """测试结果按相似度降序排列"""
        chunks = [make_chunk(f"不同主题的内容编号{i}", i) for i in range(10)]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)

        results = self.store.similarity_search_with_score(embeddings[0], k=5)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_similarity_search_k_exceeds_total(self):
        """测试 k 超过总数时返回所有"""
        chunks = [make_chunk(f"内容{i}", i) for i in range(3)]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)

        results = self.store.similarity_search(embeddings[0], k=10)
        self.assertEqual(len(results), 3)

    def test_similarity_search_empty_store(self):
        """测试空存储搜索返回空列表"""
        query_emb = self.emb.embed_query("任意查询")
        results = self.store.similarity_search(query_emb, k=5)
        self.assertEqual(len(results), 0)

    def test_filter(self):
        """测试元数据过滤"""
        chunk_a = make_chunk("人工智能文档", 0, category="A")
        chunk_b = make_chunk("烹饪美食文档", 1, category="B")
        all_chunks = [chunk_a, chunk_b]
        all_embs = embed_chunks(all_chunks)
        self.store.add_documents(all_chunks, all_embs)

        results = self.store.similarity_search(
            all_embs[0], k=5, filter={"category": "A"}
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["category"], "A")

    def test_filter_no_match(self):
        """测试过滤条件无匹配"""
        chunks = [make_chunk("测试内容", 0)]
        self.store.add_documents(chunks, embed_chunks(chunks))

        query_emb = self.emb.embed_query("测试查询")
        results = self.store.similarity_search(
            query_emb, k=5, filter={"nonexistent": "value"}
        )
        self.assertEqual(len(results), 0)

    def test_delete(self):
        """测试删除文档块"""
        chunks = [make_chunk(f"内容{i}", i) for i in range(5)]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)

        self.store.delete([chunks[0].chunk_id, chunks[1].chunk_id])
        self.assertEqual(self.store.count(), 3)

    def test_delete_nonexistent(self):
        """测试删除不存在的 ID 不报错"""
        self.store.delete(["nonexistent-id"])
        self.assertEqual(self.store.count(), 0)

    def test_clear(self):
        """测试清空所有数据"""
        chunks = [make_chunk(f"内容{i}", i) for i in range(5)]
        embeddings = embed_chunks(chunks)
        self.store.add_documents(chunks, embeddings)

        self.store.clear()
        self.assertEqual(self.store.count(), 0)
        query_emb = self.emb.embed_query("查询")
        results = self.store.similarity_search(query_emb, k=5)
        self.assertEqual(len(results), 0)

    def test_identical_vectors(self):
        """测试相同向量的相似度为 1.0"""
        chunk = make_chunk("相同向量测试", 0)
        emb = embed_chunks([chunk])
        self.store.add_documents([chunk], emb)

        results = self.store.similarity_search_with_score(emb[0], k=1)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)


# ============================================================
# ChromaVectorStore 测试
# ============================================================
class TestChromaVectorStore(unittest.TestCase):
    """ChromaVectorStore 测试"""

    def setUp(self):
        self.persist_dir = os.path.join(
            os.path.dirname(__file__), "_test_chroma_db"
        )
        self._collection_name = f"test_{uuid.uuid4().hex[:8]}"
        self.emb = get_embedding()

    def tearDown(self):
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

    def _create_store(self, persist=False):
        from rag.vectorstore.chroma_store import ChromaVectorStore

        if persist:
            return ChromaVectorStore(
                collection_name="test_persist",
                persist_directory=self.persist_dir,
            )
        return ChromaVectorStore(collection_name=self._collection_name)

    def test_add_and_count(self):
        """测试添加文档块并计数"""
        store = self._create_store()
        chunks = [make_chunk(f"这是第{i}段测试内容", i) for i in range(5)]
        embeddings = embed_chunks(chunks)
        store.add_documents(chunks, embeddings)
        self.assertEqual(store.count(), 5)

    def test_similarity_search(self):
        """测试相似度搜索"""
        store = self._create_store()
        chunks = [
            make_chunk("人工智能是计算机科学的分支", 0),
            make_chunk("机器学习使用统计方法", 1),
            make_chunk("深度学习基于神经网络", 2),
            make_chunk("今天天气真好", 3),
        ]
        embeddings = embed_chunks(chunks)
        store.add_documents(chunks, embeddings)

        query_emb = self.emb.embed_query("AI技术")
        results = store.similarity_search(query_emb, k=3)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], Document_Chunk)

    def test_similarity_search_with_score(self):
        """测试带分数的相似度搜索"""
        store = self._create_store()
        chunks = [
            make_chunk("机器学习是人工智能的子领域", 0),
            make_chunk("烹饪美食是生活的一部分", 1),
        ]
        embeddings = embed_chunks(chunks)
        store.add_documents(chunks, embeddings)

        results = store.similarity_search_with_score(embeddings[0], k=2)
        self.assertEqual(len(results), 2)
        for chunk, score in results:
            self.assertIsInstance(chunk, Document_Chunk)
            self.assertIsInstance(score, float)

    def test_delete(self):
        """测试删除文档块"""
        store = self._create_store()
        chunks = [make_chunk(f"内容{i}", i) for i in range(5)]
        embeddings = embed_chunks(chunks)
        store.add_documents(chunks, embeddings)

        store.delete([chunks[0].chunk_id])
        self.assertEqual(store.count(), 4)

    def test_clear(self):
        """测试清空所有数据"""
        store = self._create_store()
        chunks = [make_chunk(f"内容{i}", i) for i in range(5)]
        embeddings = embed_chunks(chunks)
        store.add_documents(chunks, embeddings)

        store.clear()
        self.assertEqual(store.count(), 0)

    def test_empty_search(self):
        """测试空存储搜索"""
        store = self._create_store()
        query_emb = self.emb.embed_query("任意查询")
        results = store.similarity_search(query_emb, k=5)
        self.assertEqual(len(results), 0)

    def test_persist_mode(self):
        """测试持久化模式"""
        store = self._create_store(persist=True)
        chunks = [make_chunk("持久化测试内容", 0)]
        embeddings = embed_chunks(chunks)
        store.add_documents(chunks, embeddings)
        self.assertEqual(store.count(), 1)

        # 重新打开应该能读到数据
        store2 = self._create_store(persist=True)
        self.assertEqual(store2.count(), 1)


if __name__ == "__main__":
    unittest.main()
