"""Provider-neutral records produced by the observability module."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.history import CanonicalMessage


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="python"))
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


class LLMInvokeStats(BaseModel):
    success: bool = False
    status: str = "running"
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    tool_use_prompt_tokens: int = 0
    estimated_cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None


class AgentInvokeStats(BaseModel):
    success: bool = False
    status: str = "running"
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    duration_ms: float = 0.0
    llm_calls: int = 0
    llm_errors: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    tool_duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None


class LLMInvoke(BaseModel):
    schema_version: Literal["easyagent.observability.v1"] = "easyagent.observability.v1"
    record_type: Literal["llm_invoke"] = "llm_invoke"
    invoke_id: str
    input: list[CanonicalMessage] = Field(default_factory=list)
    output: list[CanonicalMessage] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    stats: LLMInvokeStats = Field(default_factory=LLMInvokeStats)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.model_dump(mode="python"))


class AgentInvoke(BaseModel):
    schema_version: Literal["easyagent.observability.v1"] = "easyagent.observability.v1"
    record_type: Literal["agent_invoke"] = "agent_invoke"
    invoke_id: str
    parent_invoke_id: str | None = None
    agent_id: str
    query: str
    trace: list[CanonicalMessage] = Field(default_factory=list)
    output: list[CanonicalMessage] = Field(default_factory=list)
    llm_invokes: list[LLMInvoke] = Field(default_factory=list)
    stats: AgentInvokeStats = Field(default_factory=AgentInvokeStats)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.model_dump(mode="python"))


__all__ = [
    "AgentInvoke",
    "AgentInvokeStats",
    "LLMInvoke",
    "LLMInvokeStats",
    "utc_now",
]
