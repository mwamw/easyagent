"""
RAG 查询转换器单元测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag.query_transform.base import BaseQueryTransformer
from rag.query_transform.hyde import HyDETransformer
from rag.query_transform.step_back import StepBackTransformer
from core.llm import EasyLLM


# ============================================================
# 全局共享实例
# ============================================================
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = EasyLLM()
    return _llm


class FailingLLM:
    """始终抛出异常的 LLM"""

    def invoke(self, messages):
        raise RuntimeError("LLM 调用失败")


# ============================================================
# HyDETransformer 测试
# ============================================================
class TestHyDETransformer(unittest.TestCase):
    """HyDETransformer 假设性文档嵌入转换器测试"""

    @classmethod
    def setUpClass(cls):
        cls.llm = get_llm()

    def test_basic_transform(self):
        """测试基本 HyDE 转换"""
        transformer = HyDETransformer(self.llm)
        result = transformer.transform("什么是RAG?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_returns_hypothetical_doc_not_query(self):
        """测试返回的是假设性文档而不是原始查询"""
        transformer = HyDETransformer(self.llm)
        query = "什么是人工智能?"
        result = transformer.transform(query)
        # 真实 LLM 应该返回比原始查询更长的假设性文档
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_llm_failure_returns_original_query(self):
        """测试 LLM 失败时返回原始查询"""
        transformer = HyDETransformer(FailingLLM())
        query = "什么是RAG?"
        result = transformer.transform(query)
        self.assertEqual(result, query)


# ============================================================
# StepBackTransformer 测试
# ============================================================
class TestStepBackTransformer(unittest.TestCase):
    """StepBackTransformer Step-Back 转换器测试"""

    @classmethod
    def setUpClass(cls):
        cls.llm = get_llm()

    def test_basic_transform(self):
        """测试基本 Step-Back 转换"""
        transformer = StepBackTransformer(self.llm)
        result = transformer.transform("Python 3.12 有什么新特性?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_returns_abstract_question(self):
        """测试返回更抽象的问题"""
        transformer = StepBackTransformer(self.llm)
        result = transformer.transform("Raft 共识算法如何保证日志一致性?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_llm_failure_returns_original_query(self):
        """测试 LLM 失败时返回原始查询"""
        transformer = StepBackTransformer(FailingLLM())
        query = "Python 3.12 有什么新特性?"
        result = transformer.transform(query)
        self.assertEqual(result, query)


# ============================================================
# 自定义 QueryTransformer 测试
# ============================================================
class TestCustomTransformer(unittest.TestCase):
    """测试自定义查询转换器的可扩展性"""

    def test_custom_transformer(self):
        """测试继承 BaseQueryTransformer 实现自定义转换"""

        class UpperCaseTransformer(BaseQueryTransformer):
            def transform(self, query: str) -> str:
                return query.upper()

        transformer = UpperCaseTransformer()
        result = transformer.transform("hello world")
        self.assertEqual(result, "HELLO WORLD")

    def test_chain_transformers(self):
        """测试链式组合多个转换器"""

        class PrefixTransformer(BaseQueryTransformer):
            def __init__(self, prefix: str):
                self.prefix = prefix

            def transform(self, query: str) -> str:
                return self.prefix + query

        class SuffixTransformer(BaseQueryTransformer):
            def __init__(self, suffix: str):
                self.suffix = suffix

            def transform(self, query: str) -> str:
                return query + self.suffix

        query = "核心问题"
        t1 = PrefixTransformer("详细解释：")
        t2 = SuffixTransformer("？请给出例子。")

        result = t2.transform(t1.transform(query))
        self.assertEqual(result, "详细解释：核心问题？请给出例子。")


if __name__ == "__main__":
    unittest.main()
