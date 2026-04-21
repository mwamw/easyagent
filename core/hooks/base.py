"""Base hook definitions."""

from __future__ import annotations

from typing import Any, Optional

from .types import HookDecision


class BaseHook:
    """Base hook with optional lifecycle interception points."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

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
