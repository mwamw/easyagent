"""
对话历史上下文来源

将 agent 的消息历史转为 ContextItem 列表。
"""
from typing import List, Optional
from context.window import ContextItem
from context.source.base import BaseContextSource


class HistoryContextSource(BaseContextSource):
    """从对话历史提取上下文"""

    def __init__(self, history: Optional[List] = None, max_turns: int = 50):
        """
        Args:
            history: Message 列表的引用（通常是 agent.history）
            max_turns: 最多保留的消息条数
        """
        self._history = history or []
        self.max_turns = max_turns

    def set_history(self, history: List) -> None:
        """更新历史引用"""
        self._history = history

    def fetch(self, query: str, max_tokens: int = 0, **kwargs) -> List[ContextItem]:
        """将对话历史转为 ContextItem 列表

        最新消息优先级最高（线性递增 0.3→0.9）。
        """
        messages = self._history[-self.max_turns:] if self.max_turns else self._history
        if not messages:
            return []

        items = []
        n = len(messages)
        for i, msg in enumerate(messages):
            # 兼容 Message 对象和 dict
            if hasattr(msg, "to_dict"):
                role = msg.role
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = "unknown"
                content = str(msg)

            # 越新的消息优先级越高
            priority = 0.3 + 0.6 * (i / max(n - 1, 1))

            items.append(ContextItem(
                content=content,
                source="history",
                priority=priority,
                metadata={"role": role, "turn_index": i},
            ))

        return items

    @property
    def source_name(self) -> str:
        return "history"
