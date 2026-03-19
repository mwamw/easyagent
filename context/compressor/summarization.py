"""
摘要压缩器

使用 LLM 将多条上下文压缩为精炼摘要。
参考：RAPTOR (递归摘要)、LLMLingua (提示压缩)
"""
from typing import List, Any, Optional
from context.window import ContextItem
from context.compressor.base import BaseCompressor
from context.token.counter import TokenCounter
import logging

logger = logging.getLogger(__name__)


class SummarizationCompressor(BaseCompressor):
    """使用 LLM 生成摘要来压缩上下文"""

    def __init__(
        self,
        llm: Any = None,
        target_ratio: float = 0.3,
        chunk_size: int = 5,
        language: str = "zh",
    ):
        """
        Args:
            llm: EasyLLM 实例
            target_ratio: 目标压缩比（0.3 表示压缩到原文 30%）
            chunk_size: 每次送给 LLM 摘要的条目数
            language: 摘要语言
        """
        self.llm = llm
        self.target_ratio = target_ratio
        self.chunk_size = chunk_size
        self.language = language
        self._counter = TokenCounter()

    def compress(
        self,
        items: List[ContextItem],
        max_tokens: int = 0,
    ) -> List[ContextItem]:
        if not items:
            return items

        total = sum(it.token_count for it in items)
        target = max_tokens if max_tokens > 0 else int(total * self.target_ratio)

        # 如果已经在预算内，不压缩
        if total <= target:
            return items

        if self.llm is None:
            # 无 LLM 时降级到截断
            return self._truncate_fallback(items, target)

        # 分批摘要
        batches = [
            items[i:i + self.chunk_size]
            for i in range(0, len(items), self.chunk_size)
        ]

        summarized = []
        tokens_per_batch = max(target // len(batches), 50)

        for batch in batches:
            combined = "\n".join(it.content for it in batch)
            avg_priority = sum(it.priority for it in batch) / len(batch)

            summary = self._summarize(combined, tokens_per_batch)
            if summary:
                summarized.append(ContextItem(
                    content=summary,
                    source=batch[0].source,
                    priority=avg_priority,
                    token_count=self._counter.count(summary),
                    metadata={"compressed": True, "original_count": len(batch)},
                ))
            else:
                # LLM 摘要失败，保留原文最高优先级的项
                best = max(batch, key=lambda it: it.priority)
                summarized.append(best)

        return summarized

    def _summarize(self, text: str, target_tokens: int) -> Optional[str]:
        """调用 LLM 生成摘要"""
        try:
            prompt_map = {
                "zh": (
                    f"请将以下内容压缩为不超过 {target_tokens} 个 token 的精炼摘要。"
                    "保留所有关键信息和事实，去除冗余。直接输出摘要，不要添加任何前缀。\n\n"
                ),
                "en": (
                    f"Summarize the following into a concise summary of at most {target_tokens} tokens. "
                    "Keep all key facts. Output the summary directly.\n\n"
                ),
            }
            prompt = prompt_map.get(self.language, prompt_map["en"])
            messages = [{"role": "user", "content": prompt + text}]
            return self.llm.invoke(messages)
        except Exception as e:
            logger.warning("摘要压缩失败: %s", e)
            return None

    def _truncate_fallback(
        self, items: List[ContextItem], target_tokens: int
    ) -> List[ContextItem]:
        """无 LLM 时的截断降级"""
        sorted_items = sorted(items, key=lambda it: it.priority, reverse=True)
        result = []
        total = 0
        for item in sorted_items:
            if total + item.token_count > target_tokens:
                break
            result.append(item)
            total += item.token_count
        return result
