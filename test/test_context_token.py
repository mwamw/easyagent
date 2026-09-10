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
    def test_unbounded_default(self):
        self.assertIsNone(TokenBudget().max_tokens)
        self.assertIsNone(TokenBudget().remaining(100000))

    def test_total_remaining(self):
        self.assertEqual(TokenBudget(10000).remaining(2500), 7500)
        self.assertEqual(TokenBudget(10).remaining(20), 0)

    def test_no_source_allocations(self):
        self.assertFalse(hasattr(TokenBudget(), "allocations"))
        self.assertFalse(hasattr(TokenBudget(), "get_budget"))

    def test_invalid_limits(self):
        for value in (-1, True, "10"):
            with self.assertRaises(ValueError):
                TokenBudget(value)


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestTokenCounter,
            TestTokenBudget,
        ],
        title="Context Token Manual Test",
    )
    exit_with_status(ok)
