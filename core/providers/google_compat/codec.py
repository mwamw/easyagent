from __future__ import annotations

from ..openai_compat.codec import OpenAIChatCodec


class GoogleCompatCodec(OpenAIChatCodec):
    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, object]:
        return {
            "role": "function",
            "content": content,
            "tool_call_id": tool_id,
            "name": tool_name,
        }
