"""
上下文压缩器测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.window import ContextItem
from context.compressor.sliding_window import SlidingWindowCompressor
from context.compressor.token_budget import TokenBudgetCompressor
from context.compressor.selective import SelectiveCompressor
from context.compressor.summarization import SummarizationCompressor
from manual_test_runner import run_manual_tests, exit_with_status


def make_item(content, source="rag", priority=0.5, token_count=10):
    return ContextItem(
        content=content, source=source,
        priority=priority, token_count=token_count
    )


class TestSlidingWindowCompressor(unittest.TestCase):
    """SlidingWindowCompressor 测试"""

    def test_within_limit(self):
        """不超限时不裁剪"""
        comp = SlidingWindowCompressor(max_items=5)
        items = [make_item(f"项{i}") for i in range(3)]
        result = comp.compress(items)
        self.assertEqual(len(result), 3)

    def test_exceeds_limit(self):
        """超限时保留最新"""
        comp = SlidingWindowCompressor(max_items=3)
        items = [make_item(f"项{i}") for i in range(6)]
        result = comp.compress(items)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].content, "项3")
        self.assertEqual(result[2].content, "项5")

    def test_with_token_limit(self):
        """结合 token 限制"""
        comp = SlidingWindowCompressor(max_items=10)
        items = [make_item(f"项{i}", token_count=20) for i in range(5)]
        result = comp.compress(items, max_tokens=50)
        total = sum(it.token_count for it in result)
        self.assertLessEqual(total, 50)

    def test_empty_input(self):
        """空输入"""
        comp = SlidingWindowCompressor(max_items=5)
        result = comp.compress([])
        self.assertEqual(len(result), 0)


class TestTokenBudgetCompressor(unittest.TestCase):
    """TokenBudgetCompressor 测试"""

    def test_within_budget(self):
        """不超预算不裁剪"""
        comp = TokenBudgetCompressor(max_tokens=1000)
        items = [make_item(f"项{i}", token_count=10) for i in range(5)]
        result = comp.compress(items)
        self.assertEqual(len(result), 5)

    def test_exceeds_budget(self):
        """超预算按优先级裁剪"""
        comp = TokenBudgetCompressor(max_tokens=30)
        items = [
            make_item("低", priority=0.1, token_count=15),
            make_item("高", priority=0.9, token_count=15),
            make_item("中", priority=0.5, token_count=15),
        ]
        result = comp.compress(items)
        total = sum(it.token_count for it in result)
        self.assertLessEqual(total, 30)
        # 高优先级应该保留
        contents = [it.content for it in result]
        self.assertIn("高", contents)

    def test_explicit_max_tokens(self):
        """显式传入 max_tokens"""
        comp = TokenBudgetCompressor(max_tokens=100)
        items = [make_item(f"项{i}", token_count=10) for i in range(10)]
        result = comp.compress(items, max_tokens=30)
        total = sum(it.token_count for it in result)
        self.assertLessEqual(total, 30)

    def test_preserves_order(self):
        """保持原始顺序"""
        comp = TokenBudgetCompressor(max_tokens=20)
        items = [
            make_item("A", priority=0.9, token_count=10),
            make_item("B", priority=0.8, token_count=10),
            make_item("C", priority=0.1, token_count=10),
        ]
        result = comp.compress(items)
        self.assertEqual(result[0].content, "A")
        self.assertEqual(result[1].content, "B")


class TestSelectiveCompressor(unittest.TestCase):
    """SelectiveCompressor 测试"""

    def test_filters_irrelevant(self):
        """过滤不相关项"""
        comp = SelectiveCompressor(query="人工智能", threshold=0.1)
        items = [
            make_item("人工智能是计算机科学的分支"),
            make_item("今天天气真好适合出门"),
            make_item("机器学习是AI的子领域"),
        ]
        result = comp.compress(items)
        # 天气相关的应该被过滤
        self.assertLess(len(result), 3)

    def test_min_items_preserved(self):
        """至少保留 min_items 项"""
        comp = SelectiveCompressor(query="量子物理", threshold=0.9, min_items=2)
        items = [
            make_item("人工智能"),
            make_item("机器学习"),
            make_item("深度学习"),
        ]
        result = comp.compress(items)
        self.assertGreaterEqual(len(result), 2)

    def test_empty_query(self):
        """空查询不过滤"""
        comp = SelectiveCompressor(query="", threshold=0.1)
        items = [make_item("测试内容")]
        result = comp.compress(items)
        self.assertEqual(len(result), 1)

    def test_set_query(self):
        """动态更新查询"""
        comp = SelectiveCompressor(threshold=0.1)
        comp.set_query("人工智能")
        items = [
            make_item("人工智能领域"),
            make_item("美食推荐"),
        ]
        result = comp.compress(items)
        self.assertGreaterEqual(len(result), 1)


class TestSummarizationCompressor(unittest.TestCase):
    """SummarizationCompressor 测试"""

    def test_within_budget_no_compression(self):
        """预算内不压缩"""
        comp = SummarizationCompressor(target_ratio=0.3)
        items = [make_item("短文本", token_count=5)]
        result = comp.compress(items, max_tokens=100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "短文本")

    def test_no_llm_fallback(self):
        """无 LLM 时降级到截断"""
        comp = SummarizationCompressor(llm=None, target_ratio=0.3)
        items = [
            make_item("内容A", priority=0.9, token_count=20),
            make_item("内容B", priority=0.1, token_count=20),
            make_item("内容C", priority=0.5, token_count=20),
        ]
        result = comp.compress(items, max_tokens=30)
        total = sum(it.token_count for it in result)
        self.assertLessEqual(total, 30)
        # 高优先级应该保留
        contents = [it.content for it in result]
        self.assertIn("内容A", contents)

    def test_empty_input(self):
        """空输入"""
        comp = SummarizationCompressor()
        result = comp.compress([])
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestSlidingWindowCompressor,
            TestTokenBudgetCompressor,
            TestSelectiveCompressor,
            TestSummarizationCompressor,
        ],
        title="Context Compressor Manual Test",
    )
    exit_with_status(ok)
