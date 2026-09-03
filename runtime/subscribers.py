"""Built-in consumers for the unified runtime event stream."""

from __future__ import annotations

from core.callbacks import CallbackManager

from .events import RuntimeEvent, RuntimeEventType


class CallbackEventSubscriber:
    def __init__(self, callback_manager: CallbackManager):
        if not isinstance(callback_manager, CallbackManager):
            raise TypeError("callback_manager must be CallbackManager")
        self.callback_manager = callback_manager

    def __call__(self, event: RuntimeEvent) -> None:
        data = event.data
        if event.type == RuntimeEventType.AGENT_INVOKE_STARTED:
            self.callback_manager.on_agent_start(event.agent_id, str(data.get("query") or ""), invocation_id=event.invocation_id)
        elif event.type == RuntimeEventType.AGENT_INVOKE_COMPLETED:
            self.callback_manager.on_agent_end(event.agent_id, str(data.get("output") or ""), invocation_id=event.invocation_id)
        elif event.type in {RuntimeEventType.AGENT_INVOKE_FAILED, RuntimeEventType.AGENT_INVOKE_INTERRUPTED}:
            error = data.get("error")
            self.callback_manager.on_agent_end(
                event.agent_id,
                str(data.get("output") or ""),
                success=False,
                error=error if isinstance(error, Exception) else None,
                invocation_id=event.invocation_id,
            )
            if isinstance(error, Exception):
                self.callback_manager.on_error(error, context=event.type.value)
        elif event.type == RuntimeEventType.LLM_INVOKE_STARTED:
            self.callback_manager.on_llm_start(data.get("request_input"), invocation_id=event.invocation_id)
        elif event.type == RuntimeEventType.LLM_INVOKE_COMPLETED:
            self.callback_manager.on_llm_end(data.get("response") or data.get("output") or "", invocation_id=event.invocation_id)
        elif event.type == RuntimeEventType.TOOL_INVOKE_STARTED:
            self.callback_manager.on_tool_start(
                str(data.get("tool_name") or ""),
                dict(data.get("arguments") or {}),
                invocation_id=event.invocation_id,
            )
        elif event.type in {RuntimeEventType.TOOL_INVOKE_COMPLETED, RuntimeEventType.TOOL_INVOKE_FAILED}:
            error = data.get("error")
            self.callback_manager.on_tool_end(
                str(data.get("tool_name") or ""),
                str(data.get("output") or ""),
                success=event.type == RuntimeEventType.TOOL_INVOKE_COMPLETED,
                error=error if isinstance(error, Exception) else None,
                invocation_id=event.invocation_id,
            )


__all__ = ["CallbackEventSubscriber"]
