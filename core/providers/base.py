"""
Provider transport base.

Provider 只负责两件事：
1. 把 request-ready replay history 组装成底层 SDK 请求参数
2. 执行原始请求/流式请求
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional


class BaseProvider(ABC):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any,
    ):
        self._configured_provider_name = kwargs.pop("_provider_name", None)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs
        self.client = self._create_client()
        self._async_client: Any = None

    @abstractmethod
    def _create_client(self) -> Any:
        raise NotImplementedError

    def _get_async_client(self) -> Any:
        raise NotImplementedError(f"{self.provider_name} provider does not implement async client access")

    @abstractmethod
    def build_request(
        self,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def invoke_raw(self, request: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def stream_raw(self, request: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def async_invoke_raw(self, request: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def async_stream_raw(self, request: Any) -> Any:
        raise NotImplementedError

    def get_tool_schema_adapter(self) -> Any:
        from .tool_schema import create_tool_schema_adapter

        return create_tool_schema_adapter(self.provider_name)

    def build_tool_payload(self, tools: Optional[Iterable[Any]]) -> Any:
        if tools is None:
            return None
        if isinstance(tools, dict):
            return tools
        items = list(tools)
        if not items:
            return []
        if all(hasattr(item, "get_spec") and callable(getattr(item, "get_spec")) for item in items):
            return self.get_tool_schema_adapter().export_tools(items)
        return items

    def close(self) -> None:
        client = getattr(self, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        async_client = getattr(self, "_async_client", None)
        aclose = getattr(async_client, "aclose", None)
        if callable(aclose):
            await aclose()
            return
        close = getattr(async_client, "close", None)
        if callable(close):
            close()

    @property
    def provider_name(self) -> str:
        if self._configured_provider_name:
            return str(self._configured_provider_name).lower()
        return self.__class__.__name__.replace("Provider", "").lower()
