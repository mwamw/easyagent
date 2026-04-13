
from __future__ import annotations

from typing import Any, Optional


class AgentError(Exception):
    """智能体基础异常类"""
    pass


class ToolRegistryError(AgentError):
    """工具注册表相关异常"""
    pass


class ToolExecutionError(AgentError):
    """工具执行异常"""
    pass


class ToolInterruption(AgentError):
    """工具执行被中断，需要调用方接管后续流程。"""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        tool_args: Optional[dict[str, Any]] = None,
        tool_id: Optional[str] = None,
        round_number: Optional[int] = None,
        status: str = "interrupted",
        metadata: Optional[dict[str, Any]] = None,
        error_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_args = dict(tool_args or {})
        self.tool_id = tool_id
        self.round_number = round_number
        self.status = status
        self.metadata = dict(metadata or {})
        self.error_type = error_type

    def to_payload(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "tool_id": self.tool_id,
            "round_number": self.round_number,
            "status": self.status,
            "metadata": dict(self.metadata),
            "error_type": self.error_type,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ToolInterruption":
        return cls(
            str(payload.get("message", "工具执行被中断。")),
            tool_name=str(payload.get("tool_name", "")),
            tool_args=dict(payload.get("tool_args") or {}),
            tool_id=payload.get("tool_id"),
            round_number=payload.get("round_number"),
            status=str(payload.get("status", "interrupted")),
            metadata=dict(payload.get("metadata") or {}),
            error_type=payload.get("error_type"),
        )


class ToolConfirmationRequired(ToolInterruption):
    """工具需要用户确认才能继续执行。"""

    def __init__(
        self,
        *,
        tool_name: str,
        tool_args: Optional[dict[str, Any]] = None,
        tool_id: Optional[str] = None,
        round_number: Optional[int] = None,
        message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        error_type: Optional[str] = None,
    ):
        super().__init__(
            message or f"工具 '{tool_name}' 需要用户确认后才能执行。",
            tool_name=tool_name,
            tool_args=tool_args,
            tool_id=tool_id,
            round_number=round_number,
            status="needs_confirmation",
            metadata=metadata,
            error_type=error_type,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ToolConfirmationRequired":
        return cls(
            tool_name=str(payload.get("tool_name", "")),
            tool_args=dict(payload.get("tool_args") or {}),
            tool_id=payload.get("tool_id"),
            round_number=payload.get("round_number"),
            message=str(payload.get("message", "")) or None,
            metadata=dict(payload.get("metadata") or {}),
            error_type=payload.get("error_type"),
        )


class LLMInvokeError(AgentError):
    """LLM 调用异常"""
    pass


class ParameterValidationError(AgentError):
    """参数验证异常"""
    pass


class MemoryError(AgentError):
    """记忆系统异常"""
    pass


class OutputParseError(AgentError):
    """输出解析异常"""
    pass


class PromptTemplateError(AgentError):
    """提示词模板异常"""
    pass


class RetrieverError(AgentError):
    """检索器异常"""
    pass


class PlanningError(AgentError):
    """规划异常"""
    pass


class SessionError(AgentError):
    """会话持久化相关异常"""
    pass


class SessionNotFoundError(SessionError):
    """会话不存在"""
    pass


class SessionSerializationError(SessionError):
    """会话序列化或反序列化失败"""
    pass
