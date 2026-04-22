"""
Provider transport base.

Provider 只负责两件事：
1. 把 request-ready replay history 组装成底层 SDK 请求参数
2. 执行原始请求/流式请求
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional


def _usage_field(payload: Any, *keys: str) -> Any:
    for key in keys:
        if payload is None:
            return None
        if isinstance(payload, dict):
            value = payload.get(key)
        else:
            value = getattr(payload, key, None)
        if value is not None:
            return value
    return None


def _usage_int(payload: Any, *keys: str) -> Optional[int]:
    value = _usage_field(payload, *keys)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _usage_float(payload: Any, *keys: str) -> Optional[float]:
    value = _usage_field(payload, *keys)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


class ProviderResponseEnvelope:
    """Wrap a provider response/message while preserving usage metadata."""

    def __init__(
        self,
        raw: Any,
        *,
        usage: Any = None,
        usage_metadata: Any = None,
    ):
        self._raw = raw
        self.usage = usage
        self.usage_metadata = usage_metadata

    def __getattr__(self, name: str) -> Any:
        raw = object.__getattribute__(self, "_raw")
        if isinstance(raw, dict) and name in raw:
            return raw[name]
        return getattr(raw, name)

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        raw = object.__getattribute__(self, "_raw")
        if hasattr(raw, "model_dump"):
            payload = raw.model_dump(*args, **kwargs)
        elif isinstance(raw, dict):
            payload = dict(raw)
        else:
            raise AttributeError("wrapped response does not support model_dump")
        if isinstance(payload, dict):
            if self.usage is not None and payload.get("usage") is None:
                payload["usage"] = self.usage
            if self.usage_metadata is not None and payload.get("usage_metadata") is None:
                payload["usage_metadata"] = self.usage_metadata
        return payload

    def unwrap(self) -> Any:
        return self._raw


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

    def wrap_response_with_usage(
        self,
        response: Any,
        *,
        usage: Any = None,
        usage_metadata: Any = None,
    ) -> Any:
        if response is None:
            return None
        if isinstance(response, dict):
            wrapped = dict(response)
            if usage is not None:
                wrapped["usage"] = usage
            if usage_metadata is not None:
                wrapped["usage_metadata"] = usage_metadata
            return wrapped
        return ProviderResponseEnvelope(
            response,
            usage=usage,
            usage_metadata=usage_metadata,
        )

    def get_usage_from_response(self, response: Any) -> dict[str, Any]:
        return {}

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
