"""Mailbox read/ack tools for multi-agent collaboration."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from runtime import AgentRuntimeManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


MAILBOX_READ_PROMPT = """Reads runtime messages from an agent's mailbox.
- Omit agent_id to read the current agent's mailbox.
- queued means not yet read; delivered means read or injected into context; consumed means acknowledged; expired means the TTL elapsed.
- mark_delivered marks queued messages as delivered.
- Use MailboxAck after processing messages, not merely receiving them.
- include_consumed and include_expired are useful for inspecting collaboration history."""


MAILBOX_ACK_PROMPT = """Marks mailbox messages as consumed after the agent has processed them.
- Specify message_ids to acknowledge selected messages.
- Use ack_all=true only after processing all pending messages.
- Acknowledgment does not delete messages.
- Do not acknowledge unread messages or work that still needs attention."""


class MailboxReadParams(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="IMPORTANT: Omit this to read your own mailbox. Do not use a file or task name. Specify another agent ID only when inspecting that agent as its manager.")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum number of messages")
    include_consumed: bool = Field(default=False, description="Include consumed messages")
    include_expired: bool = Field(default=False, description="Include expired messages")
    mark_delivered: bool = Field(default=True, description="Mark queued messages as delivered")


class MailboxAckParams(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="IMPORTANT: Omit this when acknowledging messages in your own mailbox.")
    message_ids: list[str] = Field(default_factory=list, description="Message IDs to acknowledge")
    ack_all: bool = Field(default=False, description="Acknowledge all unconsumed messages for the current agent")


class _MailboxToolBase(Tool):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None, **kwargs):
        self.agent_runtime = agent_runtime
        self.parent_agent = parent_agent
        super().__init__(**kwargs)

    def _resolve_agent_id(self, explicit_agent_id: Optional[str]) -> str:
        candidate = str(explicit_agent_id or "").strip()
        if candidate:
            return candidate

        resolver = getattr(self.parent_agent, "_get_runtime_agent_id", None)
        if callable(resolver):
            resolved = resolver()
            if resolved:
                return str(resolved)

        execution_context = getattr(self.parent_agent, "execution_context", None)
        metadata = dict(getattr(execution_context, "metadata", {}) or {})
        fallback = metadata.get("agentId") or metadata.get("agent_id")
        if fallback:
            return str(fallback)
        raise ValueError("当前 agent 未绑定 runtime agent id，请显式传入 agent_id。")


class MailboxReadTool(_MailboxToolBase):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            parent_agent=parent_agent,
            name="MailboxRead",
            description="Reads an agent mailbox and optionally marks messages delivered.",
            parameters=MailboxReadParams,
            guidance=(
                "Inspect runtime messages and full structured payloads, including messages already injected into context."
            ),
            prompt=MAILBOX_READ_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "mailbox", "collaboration"],
            risk_categories=["side_effect"],
            side_effect_level="low",
            resource_scope=["runtime", "mailbox"],
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            agent_id = self._resolve_agent_id(parameters.get("agent_id"))
            messages = self.agent_runtime.read_mailbox(
                agent_id,
                limit=parameters.get("limit"),
                include_consumed=bool(parameters.get("include_consumed", False)),
                include_expired=bool(parameters.get("include_expired", False)),
                mark_delivered=bool(parameters.get("mark_delivered", True)),
            )
        except Exception as exc:
            return ToolResult.error(
                f"读取 mailbox 失败: {exc}",
                error_type="mailbox_read_failed",
                metadata={"agent_id": parameters.get("agent_id")},
            )

        payload = {
            "agentId": agent_id,
            "count": len(messages),
            "messages": [message.to_dict() for message in messages],
            "includeConsumed": bool(parameters.get("include_consumed", False)),
            "includeExpired": bool(parameters.get("include_expired", False)),
            "markDelivered": bool(parameters.get("mark_delivered", True)),
            "limit": int(parameters.get("limit") or 100),
        }
        return ToolResult.success(
            content=f"已读取 {len(messages)} 条 mailbox 消息",
            display_text=format_structured_display(
                f"已读取 {len(messages)} 条 mailbox 消息",
                payload,
            ),
            structured_data=payload,
            metadata=payload,
        )


class MailboxAckTool(_MailboxToolBase):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            parent_agent=parent_agent,
            name="MailboxAck",
            description="Acknowledges processed mailbox messages as consumed.",
            parameters=MailboxAckParams,
            guidance=(
                "Distinguishes message delivery from completed processing."
            ),
            prompt=MAILBOX_ACK_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "mailbox", "collaboration"],
            risk_categories=["side_effect"],
            side_effect_level="low",
            resource_scope=["runtime", "mailbox"],
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            agent_id = self._resolve_agent_id(parameters.get("agent_id"))
            actor_id = None
            resolver = getattr(self.parent_agent, "_get_runtime_agent_id", None)
            if callable(resolver):
                actor_id = resolver() or None
            if actor_id is None:
                actor_id = getattr(self.parent_agent, "name", None)
            acked = self.agent_runtime.ack_mailbox(
                agent_id,
                message_ids=list(parameters.get("message_ids") or []),
                ack_all=bool(parameters.get("ack_all", False)),
                actor_id=actor_id,
            )
        except Exception as exc:
            return ToolResult.error(
                f"确认 mailbox 消息失败: {exc}",
                error_type="mailbox_ack_failed",
                metadata={"agent_id": parameters.get("agent_id")},
            )

        payload = {
            "agentId": agent_id,
            "count": len(acked),
            "ackedAll": bool(parameters.get("ack_all", False)),
            "messageIds": list(parameters.get("message_ids") or []),
            "messages": [message.to_dict() for message in acked],
        }
        return ToolResult.success(
            content=f"已确认 {len(acked)} 条 mailbox 消息",
            display_text=format_structured_display(
                f"已确认 {len(acked)} 条 mailbox 消息",
                payload,
            ),
            structured_data=payload,
            metadata=payload,
        )


def register_mailbox_tools(
    registry: ToolRegistry,
    *,
    agent_runtime: AgentRuntimeManager,
    parent_agent: Any | None = None,
    expose_in_deferred: bool | None = True,
) -> tuple[MailboxReadTool, MailboxAckTool]:
    read_tool = MailboxReadTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    ack_tool = MailboxAckTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    registry.register_tool(read_tool, expose_in_deferred=expose_in_deferred)
    registry.register_tool(ack_tool, expose_in_deferred=expose_in_deferred)
    return read_tool, ack_tool


__all__ = [
    "MailboxAckTool",
    "MailboxReadTool",
    "register_mailbox_tools",
]
