"""Hook manager implementation."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .base import BaseHook
from .types import HookDecision, HookExecutionResult


class HookManager:
    """Runs blocking/modifying hooks for runtime lifecycle stages."""

    def __init__(self, hooks: Optional[Iterable[BaseHook]] = None):
        self._hooks: list[BaseHook] = list(hooks or [])

    @property
    def hooks(self) -> list[BaseHook]:
        return list(self._hooks)

    def add_hook(self, hook: BaseHook) -> BaseHook:
        self._hooks.append(hook)
        return hook

    def extend(self, hooks: Iterable[BaseHook]) -> None:
        for hook in hooks:
            self.add_hook(hook)

    def before_llm_request(self, payload: dict[str, Any]) -> HookExecutionResult:
        return self._run_stage("before_llm_request", payload)

    def after_llm_response(self, payload: dict[str, Any]) -> HookExecutionResult:
        return self._run_stage("after_llm_response", payload)

    def before_tool_use(self, payload: dict[str, Any]) -> HookExecutionResult:
        return self._run_stage("before_tool_use", payload)

    def after_tool_use(self, payload: dict[str, Any]) -> HookExecutionResult:
        return self._run_stage("after_tool_use", payload)

    def before_compaction(self, payload: dict[str, Any]) -> HookExecutionResult:
        return self._run_stage("before_compaction", payload)

    def after_session_restore(self, payload: dict[str, Any]) -> HookExecutionResult:
        return self._run_stage("after_session_restore", payload)

    def _run_stage(self, stage: str, payload: dict[str, Any]) -> HookExecutionResult:
        current_payload = dict(payload or {})
        audit: list[dict[str, Any]] = []
        for hook in self._hooks:
            handler = getattr(hook, stage, None)
            if not callable(handler):
                continue
            decision = handler(dict(current_payload))
            if decision is None:
                continue
            if not isinstance(decision, HookDecision):
                raise TypeError(
                    f"{hook.name}.{stage} 必须返回 HookDecision 或 None，实际收到: {type(decision).__name__}"
                )
            audit_entry = {
                "hook": hook.name,
                "stage": stage,
                "action": decision.action,
                "message": decision.message,
                "metadata": dict(decision.metadata),
            }
            audit.append(audit_entry)
            if decision.action == "block":
                return HookExecutionResult(
                    payload=current_payload,
                    audit=audit,
                    blocked=True,
                    message=decision.message,
                    error_type=decision.error_type,
                )
            if decision.updates:
                current_payload.update(decision.updates)
        return HookExecutionResult(payload=current_payload, audit=audit)
