"""Base hook definitions."""

from __future__ import annotations

from typing import Any, Optional

from .types import HookDecision


class BaseHook:
    """Base hook with optional lifecycle interception points."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def before_context(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        if type(self).abefore_context is not BaseHook.abefore_context:
            raise RuntimeError(f"{self.name} requires async execution; use ainvoke/astream")
        return None

    async def abefore_context(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        """Override for I/O. The synchronous fallback is for lightweight rules."""
        return self.before_context(payload)

    def before_llm_request(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        return None

    def after_llm_response(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        return None

    def before_tool_use(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        return None

    def after_tool_use(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        return None

    def before_compaction(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        return None

    def after_session_restore(self, payload: dict[str, Any]) -> Optional[HookDecision]:
        return None
