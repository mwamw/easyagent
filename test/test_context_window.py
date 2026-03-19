"""
ContextItem / ContextWindow 数据结构测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.window import ContextItem, ContextWindow
from context.token.counter import TokenCounter
from manual_test_runner import run_manual_tests, exit_with_status


class TestContextItem(unittest.TestCase):
    """ContextItem 测试"""

    def test_basic_creation(self):
        """基本创建"""
        item = ContextItem(content="测试内容", source="rag")
        self.assertEqual(item.content, "测试内容")
        self.assertEqual(item.source, "rag")
        self.assertEqual(item.priority, 0.5)
        self.assertGreater(item.token_count, 0)

    def test_custom_priority(self):
        """自定义优先级"""
        item = ContextItem(content="高优先级", source="system", priority=0.9)
        self.assertEqual(item.priority, 0.9)

    def test_metadata(self):
        """元数据"""
        item = ContextItem(
            content="测试", source="rag",
            metadata={"chunk_id": "abc123"}
        )
        self.assertEqual(item.metadata["chunk_id"], "abc123")

    def test_explicit_token_count(self):
        """显式指定 token 数不被覆盖"""
        item = ContextItem(content="测试", source="rag", token_count=42)
        self.assertEqual(item.token_count, 42)


class TestContextWindow(unittest.TestCase):
    """ContextWindow 测试"""

    def test_empty_window(self):
        """空窗口"""
        window = ContextWindow(max_tokens=1000)
        self.assertEqual(len(window), 0)
        self.assertEqual(window.total_tokens, 0)
        self.assertEqual(window.remaining_tokens, 1000)

    def test_add_within_budget(self):
        """预算内添加"""
        window = ContextWindow(max_tokens=10000)
        item = ContextItem(content="测试内容", source="rag", token_count=10)
        self.assertTrue(window.add(item))
        self.assertEqual(len(window), 1)
        self.assertEqual(window.total_tokens, 10)

    def test_add_exceeds_budget(self):
        """超预算添加失败"""
        window = ContextWindow(max_tokens=5)
        item = ContextItem(content="很长的文本" * 100, source="rag", token_count=100)
        self.assertFalse(window.add(item))
        self.assertEqual(len(window), 0)

    def test_add_force(self):
        """强制添加不受预算限制"""
        window = ContextWindow(max_tokens=5)
        item = ContextItem(content="超长文本", source="rag", token_count=100)
        window.add_force(item)
        self.assertEqual(len(window), 1)

    def test_remove(self):
        """移除项"""
        window = ContextWindow(max_tokens=10000)
        item = ContextItem(content="删我", source="rag", token_count=10)
        window.add(item)
        window.remove(item)
        self.assertEqual(len(window), 0)

    def test_clear(self):
        """清空"""
        window = ContextWindow(max_tokens=10000)
        for i in range(5):
            window.add(ContextItem(content=f"项{i}", source="rag", token_count=5))
        window.clear()
        self.assertEqual(len(window), 0)

    def test_items_by_source(self):
        """按来源筛选"""
        window = ContextWindow(max_tokens=10000)
        window.add(ContextItem(content="rag1", source="rag", token_count=5))
        window.add(ContextItem(content="mem1", source="memory", token_count=5))
        window.add(ContextItem(content="rag2", source="rag", token_count=5))

        rag_items = window.items_by_source("rag")
        self.assertEqual(len(rag_items), 2)
        mem_items = window.items_by_source("memory")
        self.assertEqual(len(mem_items), 1)

    def test_tokens_by_source(self):
        """按来源统计 token"""
        window = ContextWindow(max_tokens=10000)
        window.add(ContextItem(content="r1", source="rag", token_count=10))
        window.add(ContextItem(content="r2", source="rag", token_count=20))
        window.add(ContextItem(content="m1", source="memory", token_count=15))

        usage = window.tokens_by_source()
        self.assertEqual(usage["rag"], 30)
        self.assertEqual(usage["memory"], 15)

    def test_sort_by_priority(self):
        """按优先级排序"""
        window = ContextWindow(max_tokens=10000)
        window.add(ContextItem(content="低", source="rag", priority=0.1, token_count=5))
        window.add(ContextItem(content="高", source="rag", priority=0.9, token_count=5))
        window.add(ContextItem(content="中", source="rag", priority=0.5, token_count=5))

        window.sort_by_priority(descending=True)
        items = window.items
        self.assertEqual(items[0].content, "高")
        self.assertEqual(items[2].content, "低")

    def test_trim_to_budget(self):
        """裁剪到预算"""
        window = ContextWindow(max_tokens=20)
        window.add_force(ContextItem(content="低优先", source="rag", priority=0.1, token_count=10))
        window.add_force(ContextItem(content="高优先", source="rag", priority=0.9, token_count=10))
        window.add_force(ContextItem(content="中优先", source="rag", priority=0.5, token_count=10))

        removed = window.trim_to_budget()
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].content, "低优先")
        self.assertEqual(window.total_tokens, 20)

    def test_to_text(self):
        """转文本"""
        window = ContextWindow(max_tokens=10000)
        window.add(ContextItem(content="段落一", source="rag", token_count=5))
        window.add(ContextItem(content="段落二", source="rag", token_count=5))

        text = window.to_text()
        self.assertIn("段落一", text)
        self.assertIn("段落二", text)

    def test_fits_budget(self):
        """预算检查"""
        window = ContextWindow(max_tokens=100)
        window.add(ContextItem(content="已有", source="rag", token_count=80))
        self.assertTrue(window.fits_budget(20))
        self.assertFalse(window.fits_budget(21))

    def test_repr(self):
        """字符串表示"""
        window = ContextWindow(max_tokens=1000)
        self.assertIn("ContextWindow", repr(window))


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestContextItem,
            TestContextWindow,
        ],
        title="Context Window Manual Test",
    )
    exit_with_status(ok)
