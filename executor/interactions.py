"""Public-contract orchestration for default executor interruption recovery."""
from __future__ import annotations

from typing import TYPE_CHECKING

from permissions import PermissionGrant
from runtime.interactions import InteractionReply
from core.history import CanonicalBlock

if TYPE_CHECKING:
    from .base import AgentExecutionServices
    from .default import DefaultAgentExecutor


def outstanding_calls(services: AgentExecutionServices) -> list[CanonicalBlock]:
    pending: dict[str, CanonicalBlock] = {}
    for message in services.context_manager.canonical:
        for block in message.content:
            if block.type == "function_call" and block.call_id:
                pending[block.call_id] = block
            elif block.type == "function_response" and block.call_id:
                pending.pop(block.call_id, None)
    return list(pending.values())


def cancel_outstanding(services: AgentExecutionServices) -> None:
    """Close incomplete call/result pairs after cancellation, never after ASK."""
    for call in outstanding_calls(services):
        services.context_manager.append_tool_result(
            "Execution was interrupted. This call has no completed result; do not assume success.",
            call.call_id, call.name or "",
        )


async def respond(
    executor: DefaultAgentExecutor, services: AgentExecutionServices, reply: InteractionReply,
) -> None:
    runtime = services.runtime
    runtime.ensure_idle()
    pending = runtime.interruptions.get_last_interrupt()
    if not pending or reply.request_id != pending.get("request_id"):
        raise ValueError("No matching pending interaction (reply may be stale)")
    if reply.action not in ("approve", "deny", "answer"):
        raise ValueError("Unknown interaction action")
    metadata = dict(pending.get("metadata") or {})
    kind = metadata.get("interaction_type")
    if reply.action == "answer" and kind != "ask_user_question":
        raise ValueError("This interaction needs approval, not an answer")
    if kind == "ask_user_question" and reply.action != "deny" and (
        reply.action != "answer" or not reply.answer.strip()
    ):
        raise ValueError("A nonempty answer is required")
    if reply.action == "approve" and kind not in ("enter_plan_mode", "exit_plan_mode"):
        if metadata.get("permission_behavior") != "ask":
            raise NotImplementedError("This interaction has no registered approval semantics")
        if services.permission_engine is None:
            raise RuntimeError("Permission engine is unavailable")

    previous = runtime.last_invocation
    query = previous.query if previous else "Continue the current task."
    state = runtime.begin_invocation(query, stream=False)
    state.round_number = int(pending.get("round_number") or 0)
    try:
        tool_id = str(pending["tool_id"])
        tool_name = str(pending["tool_name"])
        if reply.action == "deny":
            services.context_manager.append_tool_result(
                "User declined this action. Do not repeat it without new authorization."
                + (" Reason: " + reply.answer if reply.answer else ""), tool_id, tool_name,
            )
        elif kind == "ask_user_question":
            services.context_manager.append_tool_result(
                "User answered: " + reply.answer.strip(), tool_id, tool_name,
            )
        elif kind in ("enter_plan_mode", "exit_plan_mode"):
            if services.plan is None:
                raise RuntimeError("Plan module is unavailable")
            if kind == "enter_plan_mode":
                services.plan.enter(allowed_actions=list(metadata.get("allowedActions") or []))
            else:
                services.plan.exit()
            services.context_manager.append_tool_result(
                "User approved the requested plan-mode change.", tool_id, tool_name,
            )
        else:
            grant = PermissionGrant(
                tool_name, dict(pending.get("tool_args") or {}),
                session_id=runtime.session_id, request_id=reply.request_id,
            )
            # Re-enters hooks, authorization and tool execution. An approval
            # cannot override explicit DENY or a hook-modified argument scope.
            with services.permission_engine.approval(grant):
                await executor.aexecute_tool_call(
                    services, state,
                    {"id": tool_id, "name": tool_name, "arguments": dict(pending.get("tool_args") or {})},
                )
        runtime.interruptions.clear_last_interrupt()
        # The interrupted model response may have contained multiple calls.
        # Resume each outstanding sibling without re-running completed calls.
        for call in outstanding_calls(services):
            if runtime.stop_requested:
                from core.Exception import AgentStopRequested
                raise AgentStopRequested(runtime.stop_reason or "Stopped")
            await executor.aexecute_tool_call(
                services, state,
                {"id": call.call_id, "name": call.name, "arguments": call.arguments or {}},
            )
    except BaseException as exc:
        import asyncio
        from core.Exception import AgentStopRequested
        if isinstance(exc, (AgentStopRequested, asyncio.CancelledError)):
            cancel_outstanding(services)
            runtime.interruptions.clear_last_interrupt()
        raise
    finally:
        runtime.end_invocation()
