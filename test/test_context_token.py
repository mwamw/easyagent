"""
Token 计数器与预算管理测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context.token.counter import TokenCounter
from context.token.budget import TokenBudget
from manual_test_runner import run_manual_tests, exit_with_status


class TestTokenCounter(unittest.TestCase):
    """TokenCounter 测试"""

    def setUp(self):
        self.counter = TokenCounter()

    def test_count_empty(self):
        """空文本返回 0"""
        self.assertEqual(self.counter.count(""), 0)

    def test_count_english(self):
        """英文文本计数"""
        tokens = self.counter.count("Hello world, this is a test.")
        self.assertGreater(tokens, 0)

    def test_count_chinese(self):
        """中文文本计数"""
        tokens = self.counter.count("你好世界，这是一个测试。")
        self.assertGreater(tokens, 0)

    def test_count_messages(self):
        """消息列表计数"""
        messages = [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
        ]
        tokens = self.counter.count_messages(messages)
        self.assertGreater(tokens, 0)
        # 应该比纯文本多（有角色开销）
        text_tokens = self.counter.count("你是一个助手") + self.counter.count("你好")
        self.assertGreater(tokens, text_tokens)

    def test_truncate_within_budget(self):
        """不超预算时不截断"""
        text = "短文本"
        result = self.counter.truncate(text, 1000)
        self.assertEqual(result, text)

    def test_truncate_exceeds_budget(self):
        """超预算时截断"""
        text = "这是一段非常长的文本。" * 100
        result = self.counter.truncate(text, 10)
        self.assertLess(len(result), len(text))


class TestTokenBudget(unittest.TestCase):
    """TokenBudget 测试"""

    def test_default_allocations(self):
        """默认分配比例"""
        budget = TokenBudget(max_tokens=10000)
        self.assertEqual(budget.get_budget("system"), 1000)
        self.assertEqual(budget.get_budget("history"), 3000)
        self.assertEqual(budget.get_budget("rag"), 3500)

    def test_custom_allocation(self):
        """自定义分配"""
        budget = TokenBudget(max_tokens=10000)
        budget.set_allocation("rag", 0.5)
        self.assertEqual(budget.get_budget("rag"), 5000)

    def test_unknown_source(self):
        """未知来源返回 0"""
        budget = TokenBudget()
        self.assertEqual(budget.get_budget("unknown"), 0)

    def test_remaining(self):
        """剩余 token 计算"""
        budget = TokenBudget(max_tokens=10000)
        used = {"system": 500, "history": 2000}
        self.assertEqual(budget.remaining(used), 7500)

    def test_redistribute(self):
        """重分配逻辑"""
        budget = TokenBudget(max_tokens=10000)
        used = {
            "system": 200,   # 预算 1000，剩余 800
            "history": 4000,  # 预算 3000，超额 1000
            "rag": 3500,      # 刚好
        }
        result = budget.redistribute(used)
        # system 只用了 200，多余的应该被重分配
        self.assertEqual(result["system"], 200)
        self.assertGreaterEqual(result["history"], 3000)

    def test_redistribute_no_deficit(self):
        """无超额时不影响"""
        budget = TokenBudget(max_tokens=10000)
        used = {"system": 100, "history": 100, "rag": 100}
        result = budget.redistribute(used)
        for source in used:
            self.assertEqual(result[source], used[source])


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestTokenCounter,
            TestTokenBudget,
        ],
        title="Context Token Manual Test",
    )
    exit_with_status(ok)
