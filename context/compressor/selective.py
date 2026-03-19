"""
选择性压缩器（Selective Context）

基于 TF-IDF 或关键词相关性评分，过滤掉与查询不相关的上下文项。
参考：Selective Context (Litman et al.)
"""
from typing import List, Optional
from context.window import ContextItem
from context.compressor.base import BaseCompressor


class SelectiveCompressor(BaseCompressor):
    """基于相关性评分的选择性过滤"""

    def __init__(
        self,
        query: str = "",
        threshold: float = 0.1,
        min_items: int = 1,
    ):
        """
        Args:
            query: 当前查询（用于计算相关性）
            threshold: 相关性阈值，低于此值的项被过滤
            min_items: 至少保留的项数
        """
        self.query = query
        self.threshold = threshold
        self.min_items = min_items

    def set_query(self, query: str) -> None:
        """更新当前查询"""
        self.query = query

    def compress(
        self,
        items: List[ContextItem],
        max_tokens: int = 0,
    ) -> List[ContextItem]:
        if not items or not self.query:
            return items

        # 计算每项与查询的相关性
        scored = [(item, self._relevance_score(item)) for item in items]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 保留高于阈值的项，但至少保留 min_items 项
        result = []
        total_tokens = 0

        for item, score in scored:
            if len(result) >= self.min_items and score < self.threshold:
                continue
            if max_tokens > 0 and total_tokens + item.token_count > max_tokens:
                if len(result) >= self.min_items:
                    break
            result.append(item)
            total_tokens += item.token_count

        return result

    def _relevance_score(self, item: ContextItem) -> float:
        """计算基于词重叠的相关性分数"""
        query_chars = set(self.query.lower())
        content_chars = set(item.content.lower())

        # 字符级 Jaccard（对中文友好）
        if not query_chars or not content_chars:
            return 0.0

        intersection = query_chars & content_chars
        union = query_chars | content_chars
        char_score = len(intersection) / len(union) if union else 0.0

        # 词级重叠（按空格 + 常用分隔符切分）
        query_words = set(self.query.lower().split())
        content_words = set(item.content.lower().split())
        if query_words and content_words:
            word_inter = query_words & content_words
            word_union = query_words | content_words
            word_score = len(word_inter) / len(word_union) if word_union else 0.0
        else:
            word_score = 0.0

        return char_score * 0.4 + word_score * 0.6
