"""
RAG Pipeline 管线端到端测试
"""
import unittest
import sys
import os
import uuid
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag.document import Document, Document_Chunk
from rag.chunker.fixed_chunker import FixedChunker
from rag.chunker.recursive_chunker import RecursiveCharacterChunker
from rag.vectorstore.memory_store import MemoryVectorStore
from rag.retriever.vector_retriever import VectorRetriever
from rag.pipeline import RAGPipeline
from rag.embedding.huggingface_embedding import HuggingFaceEmbedding
from core.llm import EasyLLM


# ============================================================
# 全局共享实例
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


# ============================================================
# RAGPipeline 基本功能测试
# ============================================================
class TestRAGPipelineBasic(unittest.TestCase):
    """RAGPipeline 基本功能测试"""

    @classmethod
    def setUpClass(cls):
        cls.llm = get_llm()
        cls.embedding = get_embedding()

    def setUp(self):
        self.store = MemoryVectorStore()

    def test_create_pipeline_with_defaults(self):
        """测试使用默认组件创建管线"""
        pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=self.store,
        )
        self.assertIsNotNone(pipeline.loader)
        self.assertIsNotNone(pipeline.chunker)
        self.assertIsNotNone(pipeline.retriever)
        self.assertIsNone(pipeline.query_transformer)

    def test_create_pipeline_with_custom_components(self):
        """测试使用自定义组件创建管线"""
        chunker = FixedChunker(chunk_size=200, chunk_overlap=20)
        retriever = VectorRetriever(self.store, self.embedding, k=3)

        pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=self.store,
            chunker=chunker,
            retriever=retriever,
            k=3,
        )
        self.assertIsInstance(pipeline.chunker, FixedChunker)

    def test_ingest_documents(self):
        """测试文档导入"""
        pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=self.store,
        )

        docs = [
            Document(
                document_id=str(uuid.uuid4()),
                document_path=f"/test/doc{i}.txt",
                content=f"这是第{i}篇测试文档的内容。包含一些关于人工智能的信息。" * 5,
                metadata={"source": f"/test/doc{i}.txt"},
                document_type="text",
            )
            for i in range(3)
        ]

        chunks = pipeline.ingest(docs)
        self.assertGreater(len(chunks), 0)
        self.assertGreater(self.store.count(), 0)

    def test_ingest_empty_list(self):
        """测试导入空文档列表"""
        pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=self.store,
        )
        chunks = pipeline.ingest([])
        self.assertEqual(len(chunks), 0)


# ============================================================
# RAGPipeline 查询测试
# ============================================================
class TestRAGPipelineQuery(unittest.TestCase):
    """RAGPipeline 查询测试"""

    @classmethod
    def setUpClass(cls):
        cls.llm = get_llm()
        cls.embedding = get_embedding()
        cls.store = MemoryVectorStore()
        cls.pipeline = RAGPipeline(
            llm=cls.llm,
            embedding=cls.embedding,
            vectorstore=cls.store,
            k=3,
        )

        docs = [
            Document(
                document_id=str(uuid.uuid4()),
                document_path="/test/ai.txt",
                content="人工智能是计算机科学的重要分支，研究如何让计算机模拟人类智能。" * 3,
                metadata={"source": "/test/ai.txt"},
                document_type="text",
            ),
            Document(
                document_id=str(uuid.uuid4()),
                document_path="/test/ml.txt",
                content="机器学习是人工智能的子领域，通过数据驱动的方法让计算机学习规律。" * 3,
                metadata={"source": "/test/ml.txt"},
                document_type="text",
            ),
        ]
        cls.pipeline.ingest(docs)

    def test_basic_query(self):
        """测试基本查询"""
        answer = self.pipeline.query("什么是人工智能？")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_query_with_custom_k(self):
        """测试自定义 k 值查询"""
        answer = self.pipeline.query("什么是机器学习？", k=1)
        self.assertIsInstance(answer, str)

    def test_query_empty_store(self):
        """测试空存储查询"""
        empty_pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=MemoryVectorStore(),
        )
        answer = empty_pipeline.query("任何问题")
        self.assertIn("未找到", answer)

    def test_query_with_sources(self):
        """测试带来源的查询"""
        result = self.pipeline.query_with_sources("什么是人工智能？")
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("chunks", result)
        self.assertIsInstance(result["sources"], list)
        self.assertIsInstance(result["chunks"], list)

    def test_query_with_sources_empty_store(self):
        """测试空存储的带来源查询"""
        empty_pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=MemoryVectorStore(),
        )
        result = empty_pipeline.query_with_sources("任何问题")
        self.assertIn("未找到", result["answer"])
        self.assertEqual(len(result["sources"]), 0)
        self.assertEqual(len(result["chunks"]), 0)

    def test_custom_prompt_template(self):
        """测试自定义提示词模板"""
        custom_prompt = "基于以下内容回答：\n{context}\n问题：{question}\n回答："
        pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=self.store,
            prompt_template=custom_prompt,
        )
        answer = pipeline.query("测试")
        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)


# ============================================================
# RAGPipeline 文件加载测试
# ============================================================
class TestRAGPipelineFileIngestion(unittest.TestCase):
    """RAGPipeline 文件加载测试"""

    @classmethod
    def setUpClass(cls):
        cls.llm = get_llm()
        cls.embedding = get_embedding()

    def setUp(self):
        self.store = MemoryVectorStore()
        self.pipeline = RAGPipeline(
            llm=self.llm,
            embedding=self.embedding,
            vectorstore=self.store,
        )

    def test_ingest_from_text_file(self):
        """测试从文本文件加载"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("这是一个测试文本文件的内容。\n" * 20)
            f.flush()
            tmp_path = f.name

        try:
            chunks = self.pipeline.ingest_from_path(tmp_path)
            self.assertGreater(len(chunks), 0)
            self.assertGreater(self.store.count(), 0)
        finally:
            os.unlink(tmp_path)

    def test_ingest_from_directory(self):
        """测试从目录批量加载"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(3):
                path = os.path.join(tmp_dir, f"doc{i}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"文档{i}的内容。" * 20)

            chunks = self.pipeline.ingest_from_path(tmp_dir)
            self.assertGreater(len(chunks), 0)

    def test_ingest_empty_file(self):
        """测试加载空文件"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name

        try:
            chunks = self.pipeline.ingest_from_path(tmp_path)
            self.assertEqual(len(chunks), 0)
        finally:
            os.unlink(tmp_path)


# ============================================================
# RAGPipeline 查询转换集成测试
# ============================================================
class TestRAGPipelineWithQueryTransformer(unittest.TestCase):
    """RAGPipeline 集成查询转换器测试"""

    def test_pipeline_with_query_transformer(self):
        """测试管线集成查询转换器"""
        from rag.query_transform.base import BaseQueryTransformer

        class TestTransformer(BaseQueryTransformer):
            def __init__(self):
                self.called = False

            def transform(self, query: str) -> str:
                self.called = True
                return query + " 详细解释"

        transformer = TestTransformer()
        embedding = get_embedding()
        store = MemoryVectorStore()
        llm = get_llm()

        pipeline = RAGPipeline(
            llm=llm,
            embedding=embedding,
            vectorstore=store,
            query_transformer=transformer,
        )

        docs = [
            Document(
                document_id="1",
                document_path="/test.txt",
                content="人工智能相关内容。" * 10,
                metadata={},
                document_type="text",
            )
        ]
        pipeline.ingest(docs)

        pipeline.query("什么是AI?")
        self.assertTrue(transformer.called)


# ============================================================
# RAGPipeline 端到端测试
# ============================================================
class TestRAGPipelineEndToEnd(unittest.TestCase):
    """RAGPipeline 端到端完整流程测试"""

    def test_full_pipeline_simple_mode(self):
        """测试简单模式的完整流程：导入 → 查询 → 获取回答"""
        llm = get_llm()
        embedding = get_embedding()
        store = MemoryVectorStore()

        pipeline = RAGPipeline(
            llm=llm,
            embedding=embedding,
            vectorstore=store,
        )

        docs = [
            Document(
                document_id="1",
                document_path="/docs/rag_intro.txt",
                content=(
                    "RAG（检索增强生成）是一种结合信息检索和文本生成的技术。"
                    "它通过从知识库中检索相关文档来增强大语言模型的回答质量。"
                    "RAG 可以减少幻觉并提供更准确的回答。"
                ) * 3,
                metadata={"topic": "RAG"},
                document_type="text",
            ),
            Document(
                document_id="2",
                document_path="/docs/llm_basics.txt",
                content=(
                    "大语言模型是基于 Transformer 架构的深度学习模型。"
                    "它们通过大规模语料库进行预训练，能够理解和生成自然语言。"
                ) * 3,
                metadata={"topic": "LLM"},
                document_type="text",
            ),
        ]
        chunks = pipeline.ingest(docs)
        self.assertGreater(len(chunks), 0)
        self.assertGreater(store.count(), 0)

        answer = pipeline.query("什么是RAG？")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

        result = pipeline.query_with_sources("什么是RAG？")
        self.assertIn("answer", result)
        self.assertGreater(len(result["chunks"]), 0)

    def test_full_pipeline_advanced_mode(self):
        """测试高级模式：自定义组件组合"""
        llm = get_llm()
        embedding = get_embedding()
        store = MemoryVectorStore()

        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        retriever = VectorRetriever(
            vectorstore=store,
            embedding=embedding,
            k=2,
            score_threshold=0.3,
        )

        pipeline = RAGPipeline(
            llm=llm,
            embedding=embedding,
            vectorstore=store,
            chunker=chunker,
            retriever=retriever,
            k=2,
        )

        docs = [
            Document(
                document_id="1",
                document_path="/test.txt",
                content="测试文档内容。" * 20,
                metadata={},
                document_type="text",
            )
        ]
        pipeline.ingest(docs)
        answer = pipeline.query("测试查询")
        self.assertIsInstance(answer, str)


if __name__ == "__main__":
    unittest.main()
