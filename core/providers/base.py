"""
Provider transport base.

Provider 只负责两件事：
1. 把 request-ready replay history 组装成底层 SDK 请求参数
2. 执行原始请求/流式请求
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


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
        tools: Optional[list[dict[str, Any]]] = None,
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
        return self.__class__.__name__.replace("Provider", "").lower()
