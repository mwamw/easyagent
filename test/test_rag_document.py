"""
RAG Document 数据模型单元测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.document import Document, Document_Chunk


class TestDocument(unittest.TestCase):
    """Document 数据模型测试"""

    def setUp(self):
        self.doc = Document(
            document_id="doc-001",
            document_path="/data/test.txt",
            content="这是一段测试文本内容。",
            metadata={"source": "/data/test.txt", "author": "test"},
            document_type="text",
        )

    def test_basic_fields(self):
        """测试基本字段"""
        self.assertEqual(self.doc.document_id, "doc-001")
        self.assertEqual(self.doc.document_path, "/data/test.txt")
        self.assertEqual(self.doc.content, "这是一段测试文本内容。")
        self.assertEqual(self.doc.document_type, "text")
        self.assertEqual(self.doc.metadata["author"], "test")

    def test_str_returns_content(self):
        """测试 __str__ 返回内容"""
        self.assertEqual(str(self.doc), "这是一段测试文本内容。")

    def test_repr_truncates_long_content(self):
        """测试 __repr__ 对长文本的截断"""
        long_doc = Document(
            document_id="doc-002",
            document_path="long.txt",
            content="A" * 100,
            metadata={},
            document_type="text",
        )
        repr_str = repr(long_doc)
        self.assertIn("...", repr_str)
        self.assertIn("A" * 50, repr_str)

    def test_repr_short_content(self):
        """测试 __repr__ 不截断短文本"""
        repr_str = repr(self.doc)
        self.assertNotIn("...", repr_str)

    def test_source_property(self):
        """测试 source 属性"""
        self.assertEqual(self.doc.source, "/data/test.txt")

    def test_source_property_missing(self):
        """测试 source 属性缺失时返回 None"""
        doc = Document(
            document_id="doc-003",
            document_path="test.txt",
            content="hello",
            metadata={},
            document_type="text",
        )
        self.assertIsNone(doc.source)

    def test_to_dict(self):
        """测试 to_dict 方法"""
        d = self.doc.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["document_id"], "doc-001")
        self.assertEqual(d["content"], "这是一段测试文本内容。")
        self.assertEqual(d["document_type"], "text")
        self.assertIn("source", d["metadata"])

    def test_from_dict(self):
        """测试 from_dict 方法"""
        data = {
            "document_id": "doc-010",
            "document_path": "/path/to/file",
            "content": "从字典创建的文档",
            "metadata": {"key": "value"},
            "document_type": "pdf",
        }
        doc = Document.from_dict(data)
        self.assertEqual(doc.document_id, "doc-010")
        self.assertEqual(doc.content, "从字典创建的文档")
        self.assertEqual(doc.document_type, "pdf")

    def test_from_dict_defaults(self):
        """测试 from_dict 缺失字段时的默认值"""
        doc = Document.from_dict({})
        self.assertEqual(doc.document_id, "")
        self.assertEqual(doc.content, "")
        self.assertEqual(doc.document_type, "text")
        self.assertEqual(doc.metadata, {})

    def test_roundtrip(self):
        """测试 to_dict -> from_dict 往返"""
        d = self.doc.to_dict()
        doc2 = Document.from_dict(d)
        self.assertEqual(doc2.document_id, self.doc.document_id)
        self.assertEqual(doc2.content, self.doc.content)
        self.assertEqual(doc2.document_type, self.doc.document_type)


class TestDocumentChunk(unittest.TestCase):
    """Document_Chunk 数据模型测试"""

    def setUp(self):
        self.chunk = Document_Chunk(
            document_id="doc-001",
            document_path="/data/test.txt",
            chunk_id="chunk-001",
            content="第一个文档块的内容。",
            metadata={"source": "/data/test.txt"},
            chunk_index=0,
        )

    def test_basic_fields(self):
        """测试基本字段"""
        self.assertEqual(self.chunk.document_id, "doc-001")
        self.assertEqual(self.chunk.chunk_id, "chunk-001")
        self.assertEqual(self.chunk.chunk_index, 0)
        self.assertEqual(self.chunk.content, "第一个文档块的内容。")

    def test_str_returns_content(self):
        """测试 __str__ 返回内容"""
        self.assertEqual(str(self.chunk), "第一个文档块的内容。")

    def test_repr(self):
        """测试 __repr__ 格式"""
        repr_str = repr(self.chunk)
        self.assertIn("Document_Chunk", repr_str)

    def test_to_dict(self):
        """测试 to_dict 方法"""
        d = self.chunk.to_dict()
        self.assertEqual(d["chunk_id"], "chunk-001")
        self.assertEqual(d["chunk_index"], 0)
        self.assertIn("document_id", d)

    def test_from_dict(self):
        """测试 from_dict 方法"""
        data = {
            "document_id": "doc-002",
            "document_path": "/test",
            "chunk_id": "chunk-100",
            "content": "块内容",
            "metadata": {},
            "chunk_index": 5,
        }
        chunk = Document_Chunk.from_dict(data)
        self.assertEqual(chunk.chunk_id, "chunk-100")
        self.assertEqual(chunk.chunk_index, 5)

    def test_from_dict_defaults(self):
        """测试 from_dict 缺失字段时的默认值"""
        chunk = Document_Chunk.from_dict({})
        self.assertEqual(chunk.document_id, "")
        self.assertEqual(chunk.chunk_id, "")
        self.assertEqual(chunk.chunk_index, 0)

    def test_roundtrip(self):
        """测试 to_dict -> from_dict 往返"""
        d = self.chunk.to_dict()
        chunk2 = Document_Chunk.from_dict(d)
        self.assertEqual(chunk2.chunk_id, self.chunk.chunk_id)
        self.assertEqual(chunk2.content, self.chunk.content)
        self.assertEqual(chunk2.chunk_index, self.chunk.chunk_index)


if __name__ == "__main__":
    unittest.main()
