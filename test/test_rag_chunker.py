"""
RAG 文本分块器单元测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag.document import Document, Document_Chunk
from rag.chunker.base import BaseChunker
from rag.chunker.fixed_chunker import FixedChunker
from rag.chunker.recursive_chunker import RecursiveCharacterChunker
from rag.embedding.huggingface_embedding import HuggingFaceEmbedding


def make_doc(content: str, doc_id: str = "doc-001") -> Document:
    """创建测试用文档"""
    return Document(
        document_id=doc_id,
        document_path="/test/doc.txt",
        content=content,
        metadata={"source": "/test/doc.txt"},
        document_type="text",
    )


# ============================================================
# FixedChunker 测试
# ============================================================
class TestFixedChunker(unittest.TestCase):
    """FixedChunker 固定大小分块器测试"""

    def test_basic_split(self):
        """测试基本分块"""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        doc = make_doc("A" * 250)
        chunks = chunker.split(doc)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, Document_Chunk)
            self.assertEqual(chunk.document_id, "doc-001")
            self.assertGreater(len(chunk.content), 0)

    def test_short_text_single_chunk(self):
        """测试短文本只生成一个块"""
        chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
        doc = make_doc("短文本")
        chunks = chunker.split(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "短文本")

    def test_empty_content(self):
        """测试空文本返回空列表"""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        doc = make_doc("")
        chunks = chunker.split(doc)
        self.assertEqual(len(chunks), 0)

    def test_chunk_size_respected(self):
        """测试每个块不超过 chunk_size"""
        chunk_size = 50
        chunker = FixedChunker(chunk_size=chunk_size, chunk_overlap=10)
        doc = make_doc("X" * 200)
        chunks = chunker.split(doc)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), chunk_size)

    def test_overlap(self):
        """测试相邻块有重叠"""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=30)
        doc = make_doc("A" * 50 + "B" * 50 + "C" * 50 + "D" * 50)
        chunks = chunker.split(doc)

        if len(chunks) >= 2:
            overlap_start = chunks[1].content[:30]
            first_end = chunks[0].content[-30:]
            self.assertEqual(overlap_start, first_end)

    def test_chunk_index_sequential(self):
        """测试块索引是连续的"""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        doc = make_doc("Test content. " * 30)
        chunks = chunker.split(doc)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)

    def test_invalid_overlap_raises(self):
        """测试 overlap >= chunk_size 抛出异常"""
        with self.assertRaises(ValueError):
            FixedChunker(chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            FixedChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_id_unique(self):
        """测试每个块的 chunk_id 唯一"""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        doc = make_doc("Hello world. " * 30)
        chunks = chunker.split(doc)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_metadata_inherited(self):
        """测试块继承文档的 metadata"""
        doc = make_doc("Test " * 100)
        doc.metadata["author"] = "tester"
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(doc)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["author"], "tester")


# ============================================================
# RecursiveCharacterChunker 测试
# ============================================================
class TestRecursiveCharacterChunker(unittest.TestCase):
    """RecursiveCharacterChunker 递归字符分块器测试"""

    def test_basic_split(self):
        """测试基本分块"""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        doc = make_doc("这是第一段。\n\n这是第二段。\n\n这是第三段。" * 5)
        chunks = chunker.split(doc)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Document_Chunk)

    def test_splits_on_paragraph_boundary(self):
        """测试优先在段落边界分割"""
        para1 = "第一段内容。" * 10
        para2 = "第二段内容。" * 10
        text = para1 + "\n\n" + para2
        chunker = RecursiveCharacterChunker(chunk_size=len(para1) + 10, chunk_overlap=0)
        chunks = chunker.split(make_doc(text))
        self.assertGreater(len(chunks), 1)

    def test_empty_content(self):
        """测试空文本返回空列表"""
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split(make_doc(""))
        self.assertEqual(len(chunks), 0)

    def test_short_text_single_chunk(self):
        """测试短文本只生成一个块"""
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split(make_doc("短文本"))
        self.assertEqual(len(chunks), 1)

    def test_custom_separators(self):
        """测试自定义分隔符"""
        chunker = RecursiveCharacterChunker(
            chunk_size=50, chunk_overlap=0, separators=["|||"]
        )
        text = "Part1" + "|||" + "Part2" + "|||" + "Part3"
        chunks = chunker.split(make_doc(text))
        self.assertGreater(len(chunks), 0)

    def test_invalid_overlap_raises(self):
        """测试 overlap >= chunk_size 抛出异常"""
        with self.assertRaises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    def test_chinese_text_split(self):
        """测试中文文本分块"""
        text = "人工智能是计算机科学的一个分支。" * 20
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(make_doc(text))
        self.assertGreater(len(chunks), 1)
        combined = "".join(c.content for c in chunks)
        self.assertIn("人工智能", combined)

    def test_mixed_separators(self):
        """测试混合分隔符文本"""
        text = "段落一的内容。\n\n段落二。\n句子一。句子二。句子三。" * 5
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(make_doc(text))
        self.assertGreater(len(chunks), 1)


# ============================================================
# SemanticChunker 测试（使用真实 HuggingFaceEmbedding）
# ============================================================
class TestSemanticChunker(unittest.TestCase):
    """SemanticChunker 语义分块器测试"""

    @classmethod
    def setUpClass(cls):
        """加载真实嵌入模型（全类共享，避免重复加载）"""
        cls.embedding = HuggingFaceEmbedding()

    def test_basic_split(self):
        """测试基本语义分块"""
        from rag.chunker.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(self.embedding, min_chunk_size=10)
        text = "人工智能很重要。机器学习是AI的子领域。\n深度学习使用神经网络。\n今天天气很好。我们去公园吧。"
        doc = make_doc(text)
        chunks = chunker.split(doc)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Document_Chunk)

    def test_single_sentence(self):
        """测试单句文本"""
        from rag.chunker.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(self.embedding)
        chunks = chunker.split(make_doc("只有一句话。"))
        self.assertEqual(len(chunks), 1)

    def test_empty_content(self):
        """测试空文本"""
        from rag.chunker.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(self.embedding)
        chunks = chunker.split(make_doc(""))
        self.assertEqual(len(chunks), 0)


# ============================================================
# TokenChunker 测试
# ============================================================
class TestTokenChunker(unittest.TestCase):
    """TokenChunker Token 分块器测试"""

    def test_basic_split(self):
        """测试基本 token 分块"""
        from rag.chunker.token_chunker import TokenChunker

        chunker = TokenChunker(chunk_size=20, chunk_overlap=5)
        doc = make_doc("Hello world. This is a test document for token chunking. " * 10)
        chunks = chunker.split(doc)
        self.assertGreater(len(chunks), 1)

    def test_token_count_metadata(self):
        """测试块 metadata 中包含 token_count"""
        from rag.chunker.token_chunker import TokenChunker

        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        doc = make_doc("Testing token count. " * 20)
        chunks = chunker.split(doc)
        for chunk in chunks:
            self.assertIn("token_count", chunk.metadata)
            self.assertLessEqual(chunk.metadata["token_count"], 50)

    def test_empty_content(self):
        """测试空文本"""
        from rag.chunker.token_chunker import TokenChunker

        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(make_doc(""))
        self.assertEqual(len(chunks), 0)

    def test_invalid_overlap_raises(self):
        """测试 overlap >= chunk_size 抛出异常"""
        from rag.chunker.token_chunker import TokenChunker

        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=50, chunk_overlap=50)


# ============================================================
# split_batch 通用测试
# ============================================================
class TestSplitBatch(unittest.TestCase):
    """split_batch 批量分块测试"""

    def test_batch_split_multiple_docs(self):
        """测试批量分割多个文档"""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        docs = [
            make_doc("Doc1 content. " * 10, doc_id="d1"),
            make_doc("Doc2 content. " * 10, doc_id="d2"),
            make_doc("Doc3 content. " * 10, doc_id="d3"),
        ]
        chunks = chunker.split_batch(docs)
        self.assertGreater(len(chunks), 3)
        doc_ids = set(c.document_id for c in chunks)
        self.assertEqual(doc_ids, {"d1", "d2", "d3"})

    def test_batch_split_empty_list(self):
        """测试批量分割空列表"""
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split_batch([])
        self.assertEqual(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
