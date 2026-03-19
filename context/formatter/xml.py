"""
XML 格式化器

结构化 XML 标签包裹上下文，对 Claude 等模型效果最好。
"""
from typing import List
from context.window import ContextItem
from context.formatter.base import BaseFormatter


def _escape_xml(text: str) -> str:
    """简单 XML 转义"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class XMLFormatter(BaseFormatter):
    """XML 标签结构化格式"""

    def format(self, items: List[ContextItem], source: str = "") -> str:
        if not items:
            return ""

        tag = source or "context"
        parts = [f"<{tag}>"]

        for i, item in enumerate(items, 1):
            src_attr = ""
            if "source" in item.metadata:
                src_attr = f' source="{_escape_xml(str(item.metadata["source"]))}"'
            elif "document_path" in item.metadata:
                src_attr = f' source="{_escape_xml(str(item.metadata["document_path"]))}"'

            parts.append(f'  <item index="{i}"{src_attr}>')
            parts.append(f"    {_escape_xml(item.content)}")
            parts.append("  </item>")

        parts.append(f"</{tag}>")
        return "\n".join(parts)
