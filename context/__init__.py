"""
上下文工程模块（Context Engineering）

提供模块化、可组合的上下文管理能力：
- 多源上下文收集（RAG、Memory、对话历史）
- Token 预算管理与智能压缩
- 多种格式化输出（Plain/XML/Markdown）
- 与 Agent 系统无缝集成

Quick Start:
    from context import ContextManager
    from context.source import RAGContextSource
    from context.formatter import XMLFormatter

    manager = ContextManager(max_tokens=8000)
    manager.add_source(RAGContextSource(pipeline))
    manager.set_formatter(XMLFormatter())

    context_str = manager.build_context("什么是RAG？", history=agent.history)
"""

# 核心
from .window import ContextItem, ContextWindow
from .builder import ContextBuilder
from .manager import ContextManager

# Token
from .token.counter import TokenCounter
from .token.budget import TokenBudget

# 来源
from .source.base import BaseContextSource
from .source.history_source import HistoryContextSource
from .source.rag_source import RAGContextSource
from .source.memory_source import MemoryContextSource

# 压缩器
from .compressor.base import BaseCompressor
from .compressor.sliding_window import SlidingWindowCompressor
from .compressor.token_budget import TokenBudgetCompressor
from .compressor.selective import SelectiveCompressor
from .compressor.summarization import SummarizationCompressor

# 格式化器
from .formatter.base import BaseFormatter
from .formatter.plain import PlainFormatter
from .formatter.xml import XMLFormatter
from .formatter.markdown import MarkdownFormatter

__all__ = [
    # 核心
    "ContextItem",
    "ContextWindow",
    "ContextBuilder",
    "ContextManager",
    # Token
    "TokenCounter",
    "TokenBudget",
    # 来源
    "BaseContextSource",
    "HistoryContextSource",
    "RAGContextSource",
    "MemoryContextSource",
    # 压缩器
    "BaseCompressor",
    "SlidingWindowCompressor",
    "TokenBudgetCompressor",
    "SelectiveCompressor",
    "SummarizationCompressor",
    # 格式化器
    "BaseFormatter",
    "PlainFormatter",
    "XMLFormatter",
    "MarkdownFormatter",
]
