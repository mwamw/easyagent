"""
纯文本格式化器
"""
from typing import List
from context.window import ContextItem
from context.formatter.base import BaseFormatter

# 来源名到中文标题的映射
_SOURCE_TITLES = {
    "rag": "参考资料",
    "memory": "记忆上下文",
    "history": "对话历史",
    "tool": "工具结果",
    "system": "系统信息",
}


class PlainFormatter(BaseFormatter):
    """纯文本格式，适用于通用模型"""

    def __init__(self, numbered: bool = True):
        """
        Args:
            numbered: 是否添加编号
        """
        self.numbered = numbered

    def format(self, items: List[ContextItem], source: str = "") -> str:
        if not items:
            return ""

        title = _SOURCE_TITLES.get(source, source)
        lines = []

        if title:
            lines.append(f"【{title}】")

        for i, item in enumerate(items, 1):
            if self.numbered:
                lines.append(f"{i}. {item.content}")
            else:
                lines.append(item.content)

        return "\n".join(lines)
