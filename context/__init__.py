"""Context framework contracts and built-in implementation exports."""

# 核心
from .history import ConversationHistory
from .models import (
    ContextCompactionResult,
    ContextUsage,
    ModelRequestOptions,
    ModelUsage,
    ProviderReplayHistory,
    ToolDefinitions,
)
from .default.window import ContextItem, ContextWindow
from .default.builder import ContextBuilder
from .default.manager import ContextManager

# Token
from .default.token.counter import TokenCounter
from .default.token.budget import TokenBudget

# 来源





# 压缩器
from .default.compressor.base import BaseCompressor
from .default.compressor.sliding_window import SlidingWindowCompressor
from .default.compressor.token_budget import TokenBudgetCompressor
from .default.compressor.selective import SelectiveCompressor
from .default.compressor.summarization import SummarizationCompressor
from .default.compressor.history import BaseHistoryCompactor, RuleBasedHistoryCompactor, LLMHistoryCompactor, HistoryCompactor

# 格式化器
from .default.formatter.base import BaseFormatter
from .default.formatter.plain import PlainFormatter
from .default.formatter.xml import XMLFormatter
from .default.formatter.markdown import MarkdownFormatter

__all__ = [
    # 核心
    "ContextItem",
    "ContextWindow",
    "ContextBuilder",
    "ContextManager",
    "ConversationHistory",
    "ContextCompactionResult",
    "ContextUsage",
    "ModelRequestOptions",
    "ModelUsage",
    "ProviderReplayHistory",
    "ToolDefinitions",
    # Token
    "TokenCounter",
    "TokenBudget",
    # 来源




    # 压缩器
    "BaseCompressor",
    "SlidingWindowCompressor",
    "TokenBudgetCompressor",
    "SelectiveCompressor",
    "SummarizationCompressor",
    "BaseHistoryCompactor",
    "RuleBasedHistoryCompactor",
    "LLMHistoryCompactor",
    "HistoryCompactor",
    # 格式化器
    "BaseFormatter",
    "PlainFormatter",
    "XMLFormatter",
    "MarkdownFormatter",
]

from .base import BaseContextManager
from .default.token.budget import ContextBudgetExceeded
__all__ += ["BaseContextManager", "ContextBudgetExceeded"]
