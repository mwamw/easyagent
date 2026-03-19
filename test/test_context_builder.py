"""
ContextBuilder / ContextManager 集成测试
"""
import unittest
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from context.window import ContextItem, ContextWindow
from context.builder import ContextBuilder
from context.manager import ContextManager
from context.source.base import BaseContextSource
from context.source.history_source import HistoryContextSource
from context.source.memory_source import MemoryContextSource
from context.compressor.token_budget import TokenBudgetCompressor
from context.compressor.sliding_window import SlidingWindowCompressor
from context.formatter.plain import PlainFormatter
from context.formatter.xml import XMLFormatter
from context.formatter.markdown import MarkdownFormatter
from context.token.budget import TokenBudget
from manual_test_runner import run_manual_tests, exit_with_status


# ============================================================
# 测试用的 Mock Source
# ============================================================
class MockSource(BaseContextSource):
    """测试用的模拟来源"""

    def __init__(self, name: str, items_data: list):
        self._name = name
        self._items_data = items_data

    def fetch(self, query, max_tokens=0, **kwargs):
        return [
            ContextItem(
                content=d["content"],
                source=self._name,
                priority=d.get("priority", 0.5),
                token_count=d.get("token_count", 10),
            )
            for d in self._items_data
        ]

    @property
    def source_name(self):
        return self._name


class MockMemoryItem:
    def __init__(self, content: str, importance: float = 0.5, memory_id: str = ""):
        self.content = content
        self.importance = importance
        self.id = memory_id


class MockWorkingMemory:
    def __init__(self, memories):
        self._memories = memories

    def get_all_memories(self):
        return list(self._memories)


class MockMemoryManage:
    def __init__(self, memories):
        self.memory_types = {
            "working": MockWorkingMemory(memories)
        }


# ============================================================
# ContextBuilder 测试
# ============================================================
class TestContextBuilder(unittest.TestCase):
    """ContextBuilder 测试"""

    def test_empty_builder(self):
        """无来源时返回空窗口"""
        builder = ContextBuilder()
        window = builder.build("查询")
        self.assertEqual(len(window), 0)

    def test_single_source(self):
        """单来源"""
        builder = ContextBuilder()
        source = MockSource("rag", [
            {"content": "结果一", "priority": 0.8, "token_count": 10},
            {"content": "结果二", "priority": 0.6, "token_count": 10},
        ])
        builder.add_source(source)
        window = builder.build("查询")
        self.assertEqual(len(window), 2)

    def test_multiple_sources(self):
        """多来源"""
        builder = ContextBuilder()
        builder.add_source(MockSource("rag", [
            {"content": "检索结果", "token_count": 10},
        ]))
        builder.add_source(MockSource("memory", [
            {"content": "记忆内容", "token_count": 10},
        ]))
        window = builder.build("查询")
        print(window.items)
        self.assertEqual(len(window), 2)
        sources = {it.source for it in window.items}
        self.assertIn("rag", sources)
        self.assertIn("memory", sources)

    def test_weight_affects_priority(self):
        """权重影响优先级"""
        builder = ContextBuilder()
        builder.add_source(
            MockSource("high", [{"content": "高权重", "priority": 0.5, "token_count": 10}]),
            weight=2.0,
        )
        builder.add_source(
            MockSource("low", [{"content": "低权重", "priority": 0.5, "token_count": 10}]),
            weight=0.3,
        )
        window = builder.build("查询")
        items = window.items
        # 高权重来源的优先级应该更高
        high_item = [it for it in items if it.source == "high"][0]
        low_item = [it for it in items if it.source == "low"][0]
        self.assertGreater(high_item.priority, low_item.priority)

    def test_global_compressor(self):
        """全局压缩器"""
        builder = ContextBuilder()
        builder.add_source(MockSource("rag", [
            {"content": f"项{i}", "token_count": 10, "priority": i * 0.1}
            for i in range(10)
        ]))
        builder.set_compressor(TokenBudgetCompressor(max_tokens=30))
        window = builder.build("查询")
        total = sum(it.token_count for it in window.items)
        self.assertLessEqual(total, 30)

    def test_source_level_compressor(self):
        """来源级压缩器"""
        builder = ContextBuilder()
        builder.add_source(
            MockSource("history", [
                {"content": f"消息{i}", "token_count": 10} for i in range(10)
            ]),
            compressor=SlidingWindowCompressor(max_items=3),
        )
        window = builder.build("查询")
        self.assertLessEqual(len(window), 3)

    def test_build_text(self):
        """构建文本输出"""
        builder = ContextBuilder()
        builder.add_source(MockSource("rag", [
            {"content": "检索内容一", "token_count": 10},
        ]))
        builder.add_source(MockSource("rag", [
            {"content": "检索结果", "token_count": 10},
        ]))
        builder.add_source(MockSource("memory", [
            {"content": "记忆内容", "token_count": 10},
        ]))
        builder.add_source(
            MockSource("history", [
                {"content": f"消息{i}", "token_count": 10} for i in range(10)
            ]),
        )

        builder.set_formatter(PlainFormatter())
        text = builder.build_text("查询")
        print("text1111:", text)
        self.assertIn("检索内容一", text)

    def test_build_text_xml(self):
        """XML 格式输出"""
        builder = ContextBuilder()
        builder.add_source(MockSource("rag", [
            {"content": "检索内容", "token_count": 10},
        ]))
        builder.set_formatter(XMLFormatter())
        text = builder.build_text("查询")
        self.assertIn("<rag>", text)

    def test_build_messages_history_and_system_context(self):
        """build_messages: history 保持多轮，非 history 聚合为一条 system"""
        builder = ContextBuilder()
        builder.add_source(MockSource("rag", [{"content": "检索内容", "token_count": 10}]))

        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ]

        messages = builder.build_messages(
            query="什么是RAG?",
            history=history,
            system_prompt="你是测试助手",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("你是测试助手", messages[0]["content"])
        self.assertIn("检索内容", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[-1], {"role": "user", "content": "什么是RAG?"})

    def test_budget_limits_window(self):
        """预算限制窗口大小"""
        budget = TokenBudget(max_tokens=50)
        builder = ContextBuilder(budget=budget)
        builder.add_source(MockSource("rag", [
            {"content": f"项{i}", "token_count": 20} for i in range(10)
        ]))
        window = builder.build("查询")
        self.assertLessEqual(window.total_tokens, 50)

    def test_chain_api(self):
        """链式调用 API"""
        builder = (
            ContextBuilder()
            .add_source(MockSource("rag", [{"content": "内容", "token_count": 5}]))
            .set_compressor(TokenBudgetCompressor())
            .set_formatter(XMLFormatter())
            .set_budget(TokenBudget(max_tokens=5000))
        )
        window = builder.build("查询")
        self.assertGreater(len(window), 0)

    def test_source_failure_graceful(self):
        """来源异常时优雅降级"""
        class FailingSource(BaseContextSource):
            def fetch(self, query, max_tokens=0, **kwargs):
                raise RuntimeError("来源失败")
            @property
            def source_name(self):
                return "failing"

        builder = ContextBuilder()
        builder.add_source(FailingSource())
        builder.add_source(MockSource("rag", [{"content": "正常结果", "token_count": 10}]))
        window = builder.build("查询")
        # 即使一个来源失败，另一个正常
        self.assertEqual(len(window), 1)


# ============================================================
# ContextManager 测试
# ============================================================
class TestContextManager(unittest.TestCase):
    """ContextManager 测试"""

    def test_simple_mode(self):
        """简单模式"""
        manager = ContextManager(max_tokens=4000)
        text = manager.build_context("查询", history=[])
        # 空来源应返回空（或仅历史）
        self.assertIsInstance(text, str)

    def test_with_source(self):
        """添加来源"""
        manager = ContextManager(max_tokens=4000)
        manager.add_source(MockSource("rag", [
            {"content": "检索结果", "token_count": 10},
        ]))
        text = manager.build_context("查询")
        self.assertIn("检索结果", text)

    def test_with_history(self):
        """自动注入历史"""
        manager = ContextManager(max_tokens=4000, auto_history=True)
        history = [
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ]
        text = manager.build_context("新问题", history=history)
        self.assertIn("之前的问题", text)

    def test_no_auto_history(self):
        """禁用自动历史"""
        manager = ContextManager(max_tokens=4000, auto_history=False)
        history = [{"role": "user", "content": "不该出现"}]
        text = manager.build_context("查询", history=history)
        self.assertNotIn("不该出现", text)

    def test_explicit_disable_history_in_build(self):
        """即使 auto_history=True，也可在单次构建中禁用历史注入"""
        manager = ContextManager(max_tokens=4000, auto_history=True)
        history = [{"role": "user", "content": "这段历史不应进入上下文"}]
        text = manager.build_context("查询", history=history, include_history=False)
        self.assertNotIn("这段历史不应进入上下文", text)

    def test_history_not_stale_between_calls(self):
        """上一轮 history 不应污染下一轮未传 history 的构建"""
        manager = ContextManager(max_tokens=4000, auto_history=True)
        history = [{"role": "user", "content": "第一轮历史"}]

        text1 = manager.build_context("第一次", history=history)
        self.assertIn("第一轮历史", text1)

        text2 = manager.build_context("第二次", history=None)
        self.assertNotIn("第一轮历史", text2)

    def test_set_formatter(self):
        """设置格式化器"""
        manager = ContextManager(max_tokens=4000)
        manager.add_source(MockSource("rag", [{"content": "内容", "token_count": 10}]))
        manager.set_formatter(XMLFormatter())
        text = manager.build_context("查询")
        self.assertIn("<rag>", text)

    def test_set_compressor(self):
        """设置压缩器"""
        manager = ContextManager(max_tokens=4000, auto_history=False)
        manager.add_source(MockSource("rag", [
            {"content": f"项{i}", "token_count": 10, "priority": i * 0.1}
            for i in range(20)
        ]))
        manager.set_compressor(TokenBudgetCompressor(max_tokens=50))
        text = manager.build_context("查询")
        self.assertIsInstance(text, str)

    def test_build_window(self):
        """构建 ContextWindow 对象"""
        manager = ContextManager(max_tokens=4000, auto_history=False)
        manager.add_source(MockSource("rag", [
            {"content": "结果", "token_count": 10},
        ]))
        window = manager.build_window("查询")
        self.assertIsInstance(window, ContextWindow)
        self.assertGreater(len(window), 0)

    def test_chain_api(self):
        """链式调用"""
        manager = (
            ContextManager(max_tokens=4000)
            .add_source(MockSource("rag", [{"content": "内容", "token_count": 5}]))
            .set_formatter(MarkdownFormatter())
            .set_compressor(TokenBudgetCompressor())
        )
        text = manager.build_context("查询")
        self.assertIsInstance(text, str)

    def test_build_messages_returns_multiturn_messages(self):
        """ContextManager.build_messages 返回多轮消息列表"""
        manager = ContextManager(max_tokens=4000, auto_history=True)
        manager.add_source(MockSource("rag", [{"content": "检索结果", "token_count": 10}]))

        history = [
            {"role": "user", "content": "历史问题"},
            {"role": "assistant", "content": "历史回答"},
        ]

        messages = manager.build_messages(
            "当前问题",
            history=history,
            system_prompt="系统提示",
        )

        self.assertGreaterEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("系统提示", messages[0]["content"])
        self.assertIn("检索结果", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "历史问题")
        self.assertEqual(messages[2]["content"], "历史回答")
        self.assertEqual(messages[-1]["content"], "当前问题")


# ============================================================
# 端到端集成测试
# ============================================================
class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""

    def test_full_pipeline_plain(self):
        """完整流程 - PlainFormatter"""
        manager = ContextManager(max_tokens=5000, auto_history=True)
        manager.add_source(MockSource("rag", [
            {"content": "人工智能是计算机科学的分支", "token_count": 15, "priority": 0.8},
            {"content": "机器学习使用统计方法", "token_count": 12, "priority": 0.7},
        ]))
        manager.set_formatter(PlainFormatter())

        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么能帮你的？"},
        ]

        text = manager.build_context("什么是人工智能？", history=history)
        self.assertIn("人工智能", text)
        self.assertIn("你好", text)

    def test_full_pipeline_xml(self):
        """完整流程 - XMLFormatter"""
        manager = ContextManager(max_tokens=5000, auto_history=False)
        manager.add_source(MockSource("rag", [
            {"content": "RAG是检索增强生成", "token_count": 10, "priority": 0.9},
        ]))
        manager.set_formatter(XMLFormatter())

        text = manager.build_context("什么是RAG？")
        self.assertIn("<rag>", text)
        self.assertIn("RAG是检索增强生成", text)

    def test_budget_respected(self):
        """预算不超限"""
        manager = ContextManager(max_tokens=50, auto_history=False)
        manager.add_source(MockSource("rag", [
            {"content": f"长内容{i}段", "token_count": 20} for i in range(10)
        ]))

        window = manager.build_window("查询")
        self.assertLessEqual(window.total_tokens, 50)

    def test_multi_source_priority(self):
        """多来源优先级排序"""
        manager = ContextManager(max_tokens=100, auto_history=False)
        manager.add_source(
            MockSource("rag", [{"content": "RAG高优", "priority": 0.9, "token_count": 30}]),
            weight=1.0,
        )
        manager.add_source(
            MockSource("memory", [{"content": "记忆低优", "priority": 0.3, "token_count": 30}]),
            weight=0.5,
        )

        window = manager.build_window("查询")
        items = window.items
        if len(items) >= 2:
            # 第一项应该是 RAG（优先级更高）
            self.assertEqual(items[0].source, "rag")


class TestMemoryContextSource(unittest.TestCase):
    def test_working_memory_respects_limit(self):
        memories = [
            MockMemoryItem(content=f"working-{i}", importance=0.5, memory_id=str(i))
            for i in range(10)
        ]
        mm = MockMemoryManage(memories)
        source = MemoryContextSource(memory_manage=mm, memory_types=["working"], limit=3)

        items = source.fetch("query")
        self.assertEqual(len(items), 3)
        self.assertTrue(items[-1].content.endswith("working-9"))


if __name__ == "__main__":
    ok = run_manual_tests(
        [
            TestContextBuilder,
            TestContextManager,
            TestEndToEnd,
            TestMemoryContextSource,
        ],
        title="Context Builder/Manager Manual Test",
    )
    exit_with_status(ok)
