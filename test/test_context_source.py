"""
上下文来源适配器测试
"""
import unittest
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.window import ContextItem
from context.source.history_source import HistoryContextSource
from context.source.rag_source import RAGContextSource
from context.source.memory_source import MemoryContextSource
from manual_test_runner import run_manual_tests, exit_with_status


class TestHistoryContextSource(unittest.TestCase):
    """HistoryContextSource 测试"""

    def test_empty_history(self):
        """空历史"""
        source = HistoryContextSource()
        items = source.fetch("查询")
        self.assertEqual(len(items), 0)

    def test_dict_messages(self):
        """字典格式消息"""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么能帮你的？"},
        ]
        source = HistoryContextSource(history=history)
        items = source.fetch("查询")
        print(items)
        self.assertEqual(len(items), 2)
        self.assertIn("user", items[0].content)
        self.assertIn("assistant", items[1].content)

    def test_message_objects(self):
        """Message 对象"""
        from core.Message import UserMessage, AssistantMessage
        history = [
            UserMessage("你好"),
            AssistantMessage("你好！"),
        ]
        source = HistoryContextSource(history=history)
        items = source.fetch("查询")
        self.assertEqual(len(items), 2)

    def test_max_turns(self):
        """最大轮次限制"""
        history = [{"role": "user", "content": f"消息{i}"} for i in range(20)]
        source = HistoryContextSource(history=history, max_turns=5)
        items = source.fetch("查询")
        self.assertEqual(len(items), 5)

    def test_priority_increases_with_recency(self):
        """越新的消息优先级越高"""
        history = [
            {"role": "user", "content": f"消息{i}"} for i in range(5)
        ]
        source = HistoryContextSource(history=history)
        items = source.fetch("查询")
        priorities = [it.priority for it in items]
        # 优先级应该递增
        for i in range(len(priorities) - 1):
            self.assertLessEqual(priorities[i], priorities[i + 1])

    def test_source_name(self):
        """来源名称"""
        source = HistoryContextSource()
        self.assertEqual(source.source_name, "history")

    def test_set_history(self):
        """动态更新历史"""
        source = HistoryContextSource()
        self.assertEqual(len(source.fetch("q")), 0)

        source.set_history([{"role": "user", "content": "新消息"}])
        self.assertEqual(len(source.fetch("q")), 1)


class TestRAGContextSource(unittest.TestCase):
    """RAGContextSource 测试"""

    def test_no_pipeline_or_retriever(self):
        """无 pipeline 和 retriever"""
        source = RAGContextSource()
        items = source.fetch("查询")
        self.assertEqual(len(items), 0)

    def test_with_mock_retriever(self):
        """使用模拟 retriever"""
        from rag.document import Document_Chunk

        chunks = [
            Document_Chunk(
                document_id="d1", document_path="/test.txt",
                chunk_id=str(uuid.uuid4()), content="检索结果一",
                metadata={"source": "/test.txt"}, chunk_index=0,
            ),
            Document_Chunk(
                document_id="d1", document_path="/test.txt",
                chunk_id=str(uuid.uuid4()), content="检索结果二",
                metadata={"source": "/test.txt"}, chunk_index=1,
            ),
        ]

        class MockRetriever:
            def retrieve(self, query, k=5):
                return chunks[:k]

        source = RAGContextSource(retriever=MockRetriever(), k=2)
        items = source.fetch("查询")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].source, "rag")
        self.assertIn("检索结果一", items[0].content)

    def test_priority_decreases_with_rank(self):
        """排名越靠后优先级越低"""
        from rag.document import Document_Chunk

        chunks = [
            Document_Chunk(
                document_id="d1", document_path="/t.txt",
                chunk_id=str(uuid.uuid4()), content=f"结果{i}",
                metadata={}, chunk_index=i,
            )
            for i in range(5)
        ]

        class MockRetriever:
            def retrieve(self, query, k=5):
                return chunks

        source = RAGContextSource(retriever=MockRetriever(), k=5)
        items = source.fetch("查询")
        priorities = [it.priority for it in items]
        for i in range(len(priorities) - 1):
            self.assertGreaterEqual(priorities[i], priorities[i + 1])

    def test_source_name(self):
        source = RAGContextSource()
        self.assertEqual(source.source_name, "rag")

    def test_with_mock_pipeline(self):
        """使用模拟 pipeline"""
        from rag.document import Document_Chunk

        chunks = [
            Document_Chunk(
                document_id="d1", document_path="/t.txt",
                chunk_id="c1", content="通过pipeline获取",
                metadata={}, chunk_index=0,
            )
        ]

        class MockRetriever:
            def retrieve(self, query, k=5):
                return chunks

        class MockPipeline:
            def __init__(self):
                self.retriever = MockRetriever()

        source = RAGContextSource(pipeline=MockPipeline())
        items = source.fetch("查询")
        self.assertEqual(len(items), 1)


class TestMemoryContextSource(unittest.TestCase):
    """MemoryContextSource 测试"""

    def test_no_memory_manage(self):
        """无 memory_manage"""
        source = MemoryContextSource()
        items = source.fetch("查询")
        self.assertEqual(len(items), 0)

    def test_with_working_memory(self):
        """有工作记忆"""
        class MockMemoryItem:
            def __init__(self):
                self.id = "m1"
                self.content = "工作记忆内容"
                self.importance = 0.8

        mem_item = MockMemoryItem()

        class MockWorkingMemory:
            def get_all_memories(self):
                return [mem_item]

        class MockMemoryManage:
            memory_types = {"working": MockWorkingMemory()}

        source = MemoryContextSource(memory_manage=MockMemoryManage())
        items = source.fetch("查询")
        self.assertGreater(len(items), 0)
        self.assertEqual(items[0].source, "memory")
        self.assertEqual(items[0].metadata["memory_type"], "working")

    def test_source_name(self):
        source = MemoryContextSource()
        self.assertEqual(source.source_name, "memory")

    def test_filter_memory_types(self):
        """过滤指定记忆类型"""
        class MockEpisodic:
            def search_memory(self, query, limit=5, **kwargs):
                return []

        class MockSemantic:
            def search_memory(self, query, limit=5, **kwargs):
                return []

        class MockMemoryManage:
            memory_types = {
                "episodic": MockEpisodic(),
                "semantic": MockSemantic(),
            }

        # 只检索 episodic
        source = MemoryContextSource(
            memory_manage=MockMemoryManage(),
            memory_types=["episodic"],
        )
        items = source.fetch("查询")
        # 不应出错，即使结果为空
        self.assertIsInstance(items, list)


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestHistoryContextSource,
            TestRAGContextSource,
            TestMemoryContextSource,
        ],
        title="Context Source Manual Test",
    )
    exit_with_status(ok)
