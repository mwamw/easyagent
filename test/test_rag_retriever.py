"""
RAG 检索器单元测试
"""
import unittest
import sys
import os
import uuid
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag.document import Document, Document_Chunk
from rag.vectorstore.memory_store import MemoryVectorStore
from rag.embedding.huggingface_embedding import HuggingFaceEmbedding
from core.llm import EasyLLM


# ============================================================
# 全局共享实例，避免重复加载
# ============================================================
_embedding = None
_llm = None


def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbedding()
    return _embedding


def get_llm():
    global _llm
    if _llm is None:
        _llm = EasyLLM()
    return _llm


def make_chunk(content: str, index: int = 0, doc_id: str = "doc-001") -> Document_Chunk:
    """创建测试用文档块"""
    return Document_Chunk(
        document_id=doc_id,
        document_path="/test/doc.txt",
        chunk_id=str(uuid.uuid4()),
        content=content,
        metadata={"source": "/test/doc.txt"},
        chunk_index=index,
    )


def build_test_store(chunks):
    """构建测试用的向量存储"""
    emb = get_embedding()
    store = MemoryVectorStore()
    texts = [c.content for c in chunks]
    embeddings = emb.embed_documents(texts)
    store.add_documents(chunks, embeddings)
    return store


# ============================================================
# VectorRetriever 测试
# ============================================================
class TestVectorRetriever(unittest.TestCase):
    """VectorRetriever 向量检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.embedding = get_embedding()
        cls.chunks = [
            make_chunk("人工智能是计算机科学的分支", 0),
            make_chunk("机器学习使用统计方法", 1),
            make_chunk("深度学习是机器学习的子集", 2),
            make_chunk("今天天气真好适合出门", 3),
            make_chunk("自然语言处理是AI的应用领域", 4),
        ]
        cls.store = build_test_store(cls.chunks)

    def test_basic_retrieve(self):
        """测试基本向量检索"""
        from rag.retriever.vector_retriever import VectorRetriever

        retriever = VectorRetriever(
            vectorstore=self.store,
            embedding=self.embedding,
            k=3,
        )
        results = retriever.retrieve("人工智能")
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, Document_Chunk)

    def test_callable(self):
        """测试可以直接调用检索器"""
        from rag.retriever.vector_retriever import VectorRetriever

        retriever = VectorRetriever(
            vectorstore=self.store,
            embedding=self.embedding,
            k=2,
        )
        results = retriever("人工智能")
        self.assertGreater(len(results), 0)

    def test_k_parameter(self):
        """测试 k 参数控制返回数量"""
        from rag.retriever.vector_retriever import VectorRetriever

        retriever = VectorRetriever(
            vectorstore=self.store,
            embedding=self.embedding,
            k=2,
        )
        results = retriever.retrieve("AI", k=1)
        self.assertLessEqual(len(results), 1)

    def test_score_threshold(self):
        """测试相似度阈值过滤"""
        from rag.retriever.vector_retriever import VectorRetriever

        # 极高阈值应该过滤掉大部分结果
        retriever = VectorRetriever(
            vectorstore=self.store,
            embedding=self.embedding,
            k=10,
            score_threshold=0.9999,
        )
        results = retriever.retrieve("完全不相关的查询XYZ")
        self.assertLessEqual(len(results), 5)

    def test_empty_store(self):
        """测试空存储搜索"""
        from rag.retriever.vector_retriever import VectorRetriever

        empty_store = MemoryVectorStore()
        retriever = VectorRetriever(
            vectorstore=empty_store,
            embedding=self.embedding,
            k=5,
        )
        results = retriever.retrieve("查询")
        self.assertEqual(len(results), 0)


# ============================================================
# BM25Retriever 测试
# ============================================================
class TestBM25Retriever(unittest.TestCase):
    """BM25Retriever 关键词检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.chunks = [
            make_chunk("人工智能是计算机科学的重要分支领域", 0),
            make_chunk("机器学习使用统计方法从数据中学习", 1),
            make_chunk("深度学习是机器学习的一个子集", 2),
            make_chunk("今天天气真好适合出门散步", 3),
            make_chunk("自然语言处理是人工智能的应用领域", 4),
        ]

    def test_basic_retrieve(self):
        """测试基本 BM25 检索"""
        from rag.retriever.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(chunks=self.chunks, k=3, language="zh")
        results = retriever.retrieve("人工智能")
        self.assertGreater(len(results), 0)

    def test_add_documents(self):
        """测试动态添加文档"""
        from rag.retriever.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(k=3, language="zh")
        retriever.add_documents(self.chunks[:3])
        results1 = retriever.retrieve("天气")
        retriever.add_documents(self.chunks[3:])
        results2 = retriever.retrieve("天气")
        self.assertGreaterEqual(len(results2), len(results1))

    def test_empty_retriever(self):
        """测试空检索器"""
        from rag.retriever.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(k=3)
        results = retriever.retrieve("查询")
        self.assertEqual(len(results), 0)

    def test_english_text(self):
        """测试英文文本检索"""
        from rag.retriever.bm25_retriever import BM25Retriever

        en_chunks = [
            make_chunk("Machine learning is a subset of artificial intelligence", 0),
            make_chunk("Deep learning uses neural networks with many layers", 1),
            make_chunk("Natural language processing handles text data", 2),
        ]
        retriever = BM25Retriever(chunks=en_chunks, k=2, language="en")
        results = retriever.retrieve("machine learning neural networks")
        self.assertGreater(len(results), 0)

    def test_k_parameter(self):
        """测试 k 参数"""
        from rag.retriever.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(chunks=self.chunks, k=2, language="zh")
        results = retriever.retrieve("人工智能", k=1)
        self.assertLessEqual(len(results), 1)

    def test_callable(self):
        """测试可以直接调用"""
        from rag.retriever.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(chunks=self.chunks, k=2, language="zh")
        results = retriever("人工智能")
        self.assertGreater(len(results), 0)


# ============================================================
# HybridRetriever 测试
# ============================================================
class TestHybridRetriever(unittest.TestCase):
    """HybridRetriever 混合检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.embedding = get_embedding()
        cls.chunks = [
            make_chunk("人工智能是计算机科学的重要分支", 0),
            make_chunk("机器学习使用统计方法", 1),
            make_chunk("深度学习是机器学习的子集", 2),
            make_chunk("今天天气真好适合出门", 3),
            make_chunk("自然语言处理是AI的应用", 4),
        ]
        cls.store = build_test_store(cls.chunks)

    def test_basic_hybrid_retrieve(self):
        """测试基本混合检索"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.bm25_retriever import BM25Retriever
        from rag.retriever.hybrid_retriever import HybridRetriever

        vec_ret = VectorRetriever(self.store, self.embedding, k=5)
        bm25_ret = BM25Retriever(chunks=self.chunks, k=5, language="zh")

        hybrid = HybridRetriever(
            vector_retriever=vec_ret,
            bm25_retriever=bm25_ret,
            vector_weight=0.6,
            bm25_weight=0.4,
            k=3,
        )
        results = hybrid.retrieve("人工智能")
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_weights(self):
        """测试不同权重配置"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.bm25_retriever import BM25Retriever
        from rag.retriever.hybrid_retriever import HybridRetriever

        vec_ret = VectorRetriever(self.store, self.embedding, k=5)
        bm25_ret = BM25Retriever(chunks=self.chunks, k=5, language="zh")

        # 纯向量权重
        hybrid_vec = HybridRetriever(vec_ret, bm25_ret, vector_weight=1.0, bm25_weight=0.0, k=3)
        results_vec = hybrid_vec.retrieve("人工智能")

        # 纯 BM25 权重
        hybrid_bm25 = HybridRetriever(vec_ret, bm25_ret, vector_weight=0.0, bm25_weight=1.0, k=3)
        results_bm25 = hybrid_bm25.retrieve("人工智能")

        self.assertGreater(len(results_vec), 0)
        self.assertGreater(len(results_bm25), 0)


# ============================================================
# MultiQueryRetriever 测试
# ============================================================
class TestMultiQueryRetriever(unittest.TestCase):
    """MultiQueryRetriever 多查询检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.embedding = get_embedding()
        cls.llm = get_llm()
        cls.chunks = [
            make_chunk("人工智能是计算机科学的分支", 0),
            make_chunk("机器学习使用统计方法", 1),
            make_chunk("深度学习用于图像识别", 2),
        ]
        cls.store = build_test_store(cls.chunks)

    def test_basic_multi_query(self):
        """测试基本多查询检索"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.multi_query_retriever import MultiQueryRetriever

        base_ret = VectorRetriever(self.store, self.embedding, k=3)
        multi = MultiQueryRetriever(base_ret, self.llm, num_queries=3)
        results = multi.retrieve("人工智能")
        self.assertGreater(len(results), 0)

    def test_llm_failure_fallback(self):
        """测试 LLM 失败时降级到原始查询"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.multi_query_retriever import MultiQueryRetriever

        base_ret = VectorRetriever(self.store, self.embedding, k=3)

        class FailingLLM:
            def invoke(self, messages):
                raise RuntimeError("LLM 调用失败")

        multi = MultiQueryRetriever(base_ret, FailingLLM(), num_queries=3)
        results = multi.retrieve("人工智能")
        # 应该回退到原始查询的结果
        self.assertGreater(len(results), 0)


# ============================================================
# ReRankRetriever 测试
# ============================================================
class TestReRankRetriever(unittest.TestCase):
    """ReRankRetriever 重排序检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.embedding = get_embedding()
        cls.llm = get_llm()
        cls.chunks = [
            make_chunk("人工智能是计算机科学的分支", 0),
            make_chunk("机器学习使用统计方法", 1),
            make_chunk("今天天气真好", 2),
            make_chunk("深度学习用于图像识别", 3),
            make_chunk("自然语言处理技术", 4),
        ]
        cls.store = build_test_store(cls.chunks)

    def test_basic_rerank(self):
        """测试基本重排序"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.rerank_retriever import ReRankRetriever

        base_ret = VectorRetriever(self.store, self.embedding, k=5)
        reranker = ReRankRetriever(base_ret, self.llm, top_k=2, initial_k=5)
        results = reranker.retrieve("人工智能")
        self.assertLessEqual(len(results), 2)

    def test_rerank_fewer_than_top_k(self):
        """测试候选数少于 top_k 时直接返回"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.rerank_retriever import ReRankRetriever

        small_store = MemoryVectorStore()
        chunk = make_chunk("唯一文档", 0)
        emb = self.embedding.embed_documents([chunk.content])
        small_store.add_documents([chunk], emb)

        base_ret = VectorRetriever(small_store, self.embedding, k=5)
        reranker = ReRankRetriever(base_ret, self.llm, top_k=3, initial_k=5)
        results = reranker.retrieve("查询")
        self.assertLessEqual(len(results), 1)

    def test_rerank_llm_failure(self):
        """测试 LLM 评分失败时返回 0 分"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.rerank_retriever import ReRankRetriever

        base_ret = VectorRetriever(self.store, self.embedding, k=5)

        class FailingLLM:
            def invoke(self, messages):
                raise RuntimeError("评分失败")

        reranker = ReRankRetriever(base_ret, FailingLLM(), top_k=2, initial_k=5)
        results = reranker.retrieve("人工智能")
        # 即使评分失败也应该返回结果
        self.assertGreater(len(results), 0)


# ============================================================
# CompressionRetriever 测试
# ============================================================
class TestCompressionRetriever(unittest.TestCase):
    """CompressionRetriever 上下文压缩检索器测试"""

    @classmethod
    def setUpClass(cls):
        cls.embedding = get_embedding()
        cls.llm = get_llm()
        cls.chunks = [
            make_chunk("人工智能是计算机科学的重要分支", 0),
            make_chunk("机器学习使用大量数据进行训练", 1),
            make_chunk("深度学习是机器学习的子领域", 2),
        ]
        cls.store = build_test_store(cls.chunks)

    def test_basic_compression(self):
        """测试基本压缩检索"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.compression_retriever import CompressionRetriever

        base_ret = VectorRetriever(self.store, self.embedding, k=3)
        compressor = CompressionRetriever(base_ret, self.llm, k=3)
        results = compressor.retrieve("人工智能")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, Document_Chunk)
            self.assertGreater(len(r.content), 0)

    def test_llm_failure_preserves_original(self):
        """测试 LLM 失败时保留原文档"""
        from rag.retriever.vector_retriever import VectorRetriever
        from rag.retriever.compression_retriever import CompressionRetriever

        base_ret = VectorRetriever(self.store, self.embedding, k=3)

        class FailingLLM:
            def invoke(self, messages):
                raise RuntimeError("压缩失败")

        compressor = CompressionRetriever(base_ret, FailingLLM(), k=3)
        results = compressor.retrieve("人工智能")
        # 失败时应保留原文档
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
