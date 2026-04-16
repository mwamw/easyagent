"""Tool interruption controller interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from Tool.BaseTool import ToolResult
from core.Exception import ToolConfirmationRequired, ToolInterruption


class BaseToolInterruptController(ABC):
    """Abstract controller responsible for interruption payload/state handling."""

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def get_last_interrupt(self) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def clear_last_interrupt(self) -> None:
        pass

    @abstractmethod
    def build_payload(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        round_number: int,
        tool_result: ToolResult,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def interruption_from_payload(self, payload: dict[str, Any]) -> ToolInterruption:
        pass

    @abstractmethod
    def create_interruption(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        round_number: int,
        tool_result: ToolResult,
    ) -> ToolInterruption:
        pass


class InMemoryToolInterruptController(BaseToolInterruptController):
    """Default in-memory interruption controller for BasicAgent."""

    def __init__(self):
        self._last_interrupt: Optional[dict[str, Any]] = None

    def export_state(self) -> dict[str, Any]:
        return {
            "last_tool_interrupt": self.get_last_interrupt(),
        }

    def restore_state(self, state: Optional[dict[str, Any]]) -> None:
        state = state or {}
        payload = state.get("last_tool_interrupt")
        self._last_interrupt = dict(payload) if isinstance(payload, dict) else None

    def get_last_interrupt(self) -> Optional[dict[str, Any]]:
        if self._last_interrupt is None:
            return None
        return dict(self._last_interrupt)

    def clear_last_interrupt(self) -> None:
        self._last_interrupt = None

    def build_payload(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        round_number: int,
        tool_result: ToolResult,
    ) -> dict[str, Any]:
        return {
            "message": tool_result.to_display_string(),
            "tool_name": tool_name,
            "tool_args": dict(tool_args),
            "tool_id": tool_id,
            "round_number": round_number,
            "status": tool_result.status,
            "metadata": dict(tool_result.metadata),
            "error_type": tool_result.error_type,
        }

    def interruption_from_payload(self, payload: dict[str, Any]) -> ToolInterruption:
        status = str(payload.get("status", "interrupted"))
        if status == "needs_confirmation":
            return ToolConfirmationRequired.from_payload(payload)
        return ToolInterruption.from_payload(payload)

    def create_interruption(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        round_number: int,
        tool_result: ToolResult,
    ) -> ToolInterruption:
        payload = self.build_payload(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_id=tool_id,
            round_number=round_number,
            tool_result=tool_result,
        )
        self._last_interrupt = dict(payload)
        return self.interruption_from_payload(payload)
