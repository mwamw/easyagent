"Request assembly, total input budgets, and replaceable compression."

# 核心
from .window import ContextItem, ContextWindow
from .builder import ContextBuilder
from .manager import ContextManager

# Token
from .token.counter import TokenCounter
from .token.budget import TokenBudget

# 来源





# 压缩器
from .compressor.base import BaseCompressor
from .compressor.sliding_window import SlidingWindowCompressor
from .compressor.token_budget import TokenBudgetCompressor
from .compressor.selective import SelectiveCompressor
from .compressor.summarization import SummarizationCompressor
from .compressor.history import BaseHistoryCompactor, RuleBasedHistoryCompactor, LLMHistoryCompactor, HistoryCompactor

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
from .manager import HistoryCompactionResult
from .token.budget import ContextBudgetExceeded
__all__ += ["BaseContextManager", "HistoryCompactionResult", "ContextBudgetExceeded"]
