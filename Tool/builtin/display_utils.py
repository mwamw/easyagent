"""Display helpers for tools that need both summaries and structured payloads."""

from __future__ import annotations

import json
from typing import Any


def dump_tool_payload(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def format_structured_display(
    summary: str,
    payload: Any,
    *,
    result_text: str | None = None,
    payload_label: str = "结构化返回",
) -> str:
    sections: list[str] = [str(summary or "").strip()]
    clean_result = str(result_text or "").strip()
    if clean_result:
        sections.append(f"结果正文:\n{clean_result}")
    sections.append(f"{payload_label}:\n{dump_tool_payload(payload)}")
    return "\n\n".join(section for section in sections if section)


def format_error_display(message: str, metadata: Any | None = None) -> str:
    clean_message = str(message or "").strip()
    if metadata in (None, {}, []):
        return clean_message
    return f"{clean_message}\n\n上下文:\n{dump_tool_payload(metadata)}"


__all__ = [
    "dump_tool_payload",
    "format_error_display",
    "format_structured_display",
]
