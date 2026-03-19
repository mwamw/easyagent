"""
Markdown 格式化器
"""
from typing import List
from context.window import ContextItem
from context.formatter.base import BaseFormatter

_SOURCE_TITLES = {
    "rag": "参考资料",
    "memory": "记忆上下文",
    "history": "对话历史",
    "tool": "工具结果",
    "system": "系统信息",
}


class MarkdownFormatter(BaseFormatter):
    """Markdown 格式，适合 GPT / Gemini 等模型"""

    def __init__(self, heading_level: int = 2):
        self.heading_level = heading_level

    def format(self, items: List[ContextItem], source: str = "") -> str:
        if not items:
            return ""

        title = _SOURCE_TITLES.get(source, source)
        prefix = "#" * self.heading_level
        lines = []

        if title:
            lines.append(f"{prefix} {title}")
            lines.append("")

        for item in items:
            src_info = ""
            src_value = item.metadata.get("source") or item.metadata.get("document_path")
            if src_value:
                src_info = f" _(来源: {src_value})_"
            lines.append(f"- {item.content}{src_info}")

        return "\n".join(lines)
