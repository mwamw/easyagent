from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Optional

from ...history import CanonicalMessage, _generic_canonical_messages_from_history_entry
from ...replay_converter import canonical_to_replay_history


class BaseProviderCodec(ABC):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return _generic_canonical_messages_from_history_entry(message, provider_name=self.provider_name)

    def prepare_messages(self, messages: list[Any]) -> list[Any]:
        prepared: list[Any] = []
        for message in messages:
            if self.is_request_ready_message(message):
                self._append_prepared(prepared, self.clone_message(message))
                continue
            converted = canonical_to_replay_history([message], self.provider_name)
            for item in converted:
                self._append_prepared(prepared, item)
        return prepared

    def _append_prepared(self, prepared: list[Any], item: Any) -> None:
        prepared.append(item)

    @staticmethod
    def clone_message(message: Any) -> Any:
        if isinstance(message, dict):
            return json.loads(json.dumps(message, ensure_ascii=False, default=str))
        return message

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

    def stream_events(self, raw_stream: Any, *, tools: bool = False) -> Generator[dict[str, Any], None, None]:
        raise NotImplementedError(f"{self.provider_name} codec does not implement sync stream parsing")

    async def astream_events(self, raw_stream: Any, *, tools: bool = False) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError(f"{self.provider_name} codec does not implement async stream parsing")
