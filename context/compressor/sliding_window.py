"""
滑动窗口压缩器

保留最近的 N 条消息，丢弃旧的消息。
适合对话历史的简单截断。
"""
from typing import List
from context.window import ContextItem
from context.compressor.base import BaseCompressor


class SlidingWindowCompressor(BaseCompressor):
    """保留最近 N 条 ContextItem"""

    def __init__(self, max_items: int = 20):
        """
        Args:
            max_items: 最多保留的条目数
        """
        self.max_items = max_items

    def compress(
        self,
        items: List[ContextItem],
        max_tokens: int = 0,
    ) -> List[ContextItem]:
        # 先按条数限制
        kept = items[-self.max_items:] if len(items) > self.max_items else list(items)

        # 再按 token 限制（从最新开始保留）
        if max_tokens > 0:
            result = []
            total = 0
            for item in reversed(kept):
                if total + item.token_count > max_tokens:
                    break
                result.insert(0, item)
                total += item.token_count
            return result

        return kept
