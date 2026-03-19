"""
Token 预算压缩器

当总 token 超出预算时，按优先级从低到高移除项。
这是最通用的压缩策略。
"""
from typing import List
from context.window import ContextItem
from context.compressor.base import BaseCompressor


class TokenBudgetCompressor(BaseCompressor):
    """按优先级裁剪，确保总 token 不超预算"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def compress(
        self,
        items: List[ContextItem],
        max_tokens: int = 0,
    ) -> List[ContextItem]:
        budget = max_tokens if max_tokens > 0 else self.max_tokens
        total = sum(it.token_count for it in items)

        if total <= budget:
            return items

        # 按优先级排序（低优先级在前），逐个移除
        sorted_items = sorted(items, key=lambda it: it.priority)
        removed_indices = set()

        for i, item in enumerate(sorted_items):
            if total <= budget:
                break
            total -= item.token_count
            removed_indices.add(id(item))

        # 保持原始顺序返回
        return [it for it in items if id(it) not in removed_indices]
