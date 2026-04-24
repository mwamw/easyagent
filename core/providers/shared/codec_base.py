from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Optional

from ...history import (
    CanonicalBlock,
    CanonicalMessage,
    _generic_canonical_messages_from_history_entry,
    _json_safe,
)


class BaseProviderCodec(ABC):
    _TOKEN_ESTIMATE_SIGNATURE_KEYS = {
        "signature",
        "thought_signature",
        "thoughtSignature",
    }

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    @classmethod
    def _strip_token_estimate_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._strip_token_estimate_metadata(item)
                for key, item in value.items()
                if key not in cls._TOKEN_ESTIMATE_SIGNATURE_KEYS
            }
        if isinstance(value, list):
            return [cls._strip_token_estimate_metadata(item) for item in value]
        if isinstance(value, tuple):
            return [cls._strip_token_estimate_metadata(item) for item in value]
        return value

    @staticmethod
    def _provider_replay_scope(provider_name: Optional[str]) -> Optional[str]:
        normalized = (provider_name or "").lower()
        if normalized in {"google_native", "gemini_native"}:
            return "google_native"
        if normalized in {"anthropic_native", "claude_native"}:
            return "anthropic_native"
        if normalized in {"google", "gemini"}:
            return "google"
        if normalized in {"anthropic", "claude"}:
            return "anthropic"
        return normalized or None

    def _canonical_origin_matches_current_provider(self, message: CanonicalMessage) -> bool:
        """Whether provider-specific payload/signatures can be replayed as-is."""
        return bool(
            message.provider
            and self._provider_replay_scope(message.provider) == self._provider_replay_scope(self.provider_name)
        )

    def _provider_payload_for_current_provider(
        self,
        message: CanonicalMessage,
        block: CanonicalBlock,
    ) -> Optional[dict[str, Any]]:
        if not self._canonical_origin_matches_current_provider(message):
            return None
        return block.payload if isinstance(block.payload, dict) else None

    def _signature_for_current_provider(
        self,
        message: CanonicalMessage,
        block: CanonicalBlock,
    ) -> Any:
        if not self._canonical_origin_matches_current_provider(message):
            return None
        return block.signature

    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _generic_canonical_messages_from_history_entry(message, provider_name=self.provider_name)

    def query_to_canonical(self, query: str) -> list[CanonicalMessage]:
        return [
            CanonicalMessage(
                role="user",
                content=[CanonicalBlock(type="text", text=query)],
                provider=self.provider_name,
                provider_message_type="user",
            )
        ]

    def query_to_replay(self, query: str) -> list[Any]:
        return self.canonical_to_replay(self.query_to_canonical(query))

    def response_to_replay(self, response: Any, *, include_reasoning: bool = False) -> list[Any]:
        payload = self.build_assistant_response(response, include_reasoning=include_reasoning)
        if payload is None:
            return []
        if isinstance(payload, list):
            return [self.clone_message(item) for item in payload]
        return [self.clone_message(payload)]

    def response_to_canonical(self, response: Any, *, include_reasoning: bool = False) -> list[CanonicalMessage]:
        entries: list[CanonicalMessage] = []
        for item in self.response_to_replay(response, include_reasoning=include_reasoning):
            entries.extend(self.history_entry_to_canonical(item))
        return entries

    def assistant_message_to_replay(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[Any]:
        payload = self.build_assistant_message(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
        )
        if payload is None:
            return []
        if isinstance(payload, list):
            return [self.clone_message(item) for item in payload]
        return [self.clone_message(payload)]

    def assistant_message_to_canonical(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[CanonicalMessage]:
        entries: list[CanonicalMessage] = []
        for item in self.assistant_message_to_replay(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
        ):
            entries.extend(self.history_entry_to_canonical(item))
        return entries

    def tool_result_to_replay(self, content: str, tool_id: str, tool_name: str) -> list[Any]:
        payload = self.build_tool_result(content, tool_id, tool_name)
        if payload is None:
            return []
        if isinstance(payload, list):
            return [self.clone_message(item) for item in payload]
        return [self.clone_message(payload)]

    def tool_result_to_canonical(self, content: str, tool_id: str, tool_name: str) -> list[CanonicalMessage]:
        entries: list[CanonicalMessage] = []
        for item in self.tool_result_to_replay(content, tool_id, tool_name):
            entries.extend(self.history_entry_to_canonical(item))
        return entries

    def replay_to_canonical(self, messages: list[Any]) -> list[CanonicalMessage]:
        entries: list[CanonicalMessage] = []
        for message in messages:
            entries.extend(self.history_entry_to_canonical(message))
        return entries

    def canonical_to_replay(self, messages: list[Any]) -> list[Any]:
        prepared: list[Any] = []
        for message in messages:
            for item in self.canonical_message_to_replay(message):
                self.append_replay_entry(prepared, item)
        return prepared

    def prepare_messages(self, messages: list[Any]) -> list[Any]:
        prepared: list[Any] = []
        for message in messages:
            if self.is_request_ready_message(message):
                self.append_replay_entry(prepared, self.clone_message(message))
                continue
            converted = self.canonical_to_replay([message])
            for item in converted:
                self.append_replay_entry(prepared, item)
        return prepared

    def append_replay_entry(self, prepared: list[Any], item: Any) -> None:
        prepared.append(item)

    @staticmethod
    def clone_message(message: Any) -> Any:
        if isinstance(message, dict):
            return _json_safe(message)
        return message

    def canonical_message_to_replay(self, message: Any) -> list[Any]:
        raise NotImplementedError

    def build_request_token_payload(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        pending_messages: Optional[list[Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "provider": self.provider_name,
            "history": [_json_safe(item) for item in replay_history],
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = _json_safe(tools)
        if pending_messages:
            payload["pending_messages"] = [_json_safe(item) for item in pending_messages]
        if reasoning:
            payload["reasoning"] = _json_safe(reasoning)
        return payload

    def count_request_tokens(
        self,
        counter: Any,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        pending_messages: Optional[list[Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> int:
        payload = self.build_request_token_payload(
            replay_history,
            system_prompt=system_prompt,
            tools=tools,
            pending_messages=pending_messages,
            reasoning=reasoning,
        )
        return counter.count(self._strip_token_estimate_metadata(payload))

    @abstractmethod
    def is_request_ready_message(self, message: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_assistant_message(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_assistant_response(self, response: Any, include_reasoning: bool = False) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_tool_result(self, content: str, tool_id: str, tool_name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_response_content(self, response: Any) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_thinking_content(self, response: Any) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def has_tool_calls(self, response: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_tool_calls(self, response: Any) -> list[Any]:
        raise NotImplementedError

    def response_has_tool_calls(self, response: Any) -> bool:
        return self.has_tool_calls(response)

    def response_tool_calls(self, response: Any) -> list[Any]:
        return self.get_tool_calls(response)

    def response_text(self, response: Any) -> Optional[str]:
        return self.get_response_content(response)

    def response_reasoning(self, response: Any) -> Optional[str]:
        return self.get_thinking_content(response)

    def stream_events(self, raw_stream: Any, *, tools: bool = False) -> Generator[dict[str, Any], None, None]:
        raise NotImplementedError(f"{self.provider_name} codec does not implement sync stream parsing")

    async def astream_events(self, raw_stream: Any, *, tools: bool = False) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError(f"{self.provider_name} codec does not implement async stream parsing")
