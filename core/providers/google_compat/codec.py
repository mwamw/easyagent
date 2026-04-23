from __future__ import annotations

from ..openai_compat.codec import OpenAIChatCodec


class GoogleCompatCodec(OpenAIChatCodec):
    def canonical_message_to_replay(self, message):
        canonical = self.history_entry_to_canonical(message)[0] if not hasattr(message, "role") and not (
            isinstance(message, dict) and message.get("record_type") == "canonical_message"
        ) else message
        from ...history import coerce_canonical_message

        coerced = coerce_canonical_message(canonical)
        if coerced is None:
            entries = []
            for entry in self.history_entry_to_canonical(message):
                entries.extend(self.canonical_message_to_replay(entry))
            return entries
        from ..openai_compat.codec import _canonical_to_openai_like_message

        return _canonical_to_openai_like_message(
            coerced,
            tool_role="function",
            preserve_reasoning=True,
            provider_payload_allowed=self._canonical_origin_matches_current_provider(coerced),
        )

    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> dict[str, object]:
        return {
            "role": "function",
            "content": content,
            "tool_call_id": tool_id,
            "name": tool_name,
        }
