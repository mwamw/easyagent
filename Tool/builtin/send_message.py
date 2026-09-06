"""Structured mailbox messaging for agent runtime collaboration."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from runtime import AgentRuntimeManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


class SendMessageParams(BaseModel):
    recipient_type: Literal["agent", "team", "task"] = Field(description="Message recipient type")
    recipient_id: str = Field(description="Recipient ID; team recipients may also use a team name")
    content: str = Field(description="Structured text message to send")
    sender_id: Optional[str] = Field(default=None, description="Sender identity; defaults to the current agent name")
    ttl_ms: Optional[int] = Field(default=None, ge=0, description="Message TTL in milliseconds; elapsed messages are marked expired")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


SEND_MESSAGE_PROMPT = """Sends a structured runtime message to an agent, team, or task scope.
- Set recipient_type to agent, team, or task and use the corresponding real recipient_id.
- For a team, recipient_id may be its ID or name.
- Write an actionable message with relevant context, constraints, and expected follow-up.
- Omit sender_id to use the current agent's identity.
- ttl_ms limits how long the message remains applicable.
- Messages are stored in mailboxes. Delivery does not mean the recipient has completed the requested work.
- This tool does not launch or resume agents. Use Agent to start new work and AgentGet or AgentWait to inspect completion."""


class SendMessageTool(Tool):
    def __init__(
        self,
        *,
        agent_runtime: AgentRuntimeManager,
        parent_agent: Any | None = None,
    ):
        self.agent_runtime = agent_runtime
        self.parent_agent = parent_agent
        super().__init__(
            name="SendMessage",
            description="Sends a structured runtime message to an agent, team, or task mailbox.",
            parameters=SendMessageParams,
            guidance=(
                "Use for constraints, context, or next actions. Delivery does not imply processing."
            ),
            prompt=SEND_MESSAGE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
            side_effect_level="medium",
            resource_scope=["runtime", "mailbox", "task"],
        )

    def run(self, parameters: dict) -> ToolResult:
        sender_id = parameters.get("sender_id") or getattr(self.parent_agent, "name", None)
        metadata = dict(parameters.get("metadata") or {})
        current_task_id = getattr(self.parent_agent, "current_task_id", None)
        if current_task_id and "taskId" not in metadata:
            metadata["taskId"] = current_task_id
        try:
            deliveries = self.agent_runtime.send_message(
                recipient_type=parameters["recipient_type"],
                recipient_id=parameters["recipient_id"],
                content=parameters["content"],
                sender_id=sender_id,
                metadata=metadata,
                ttl_ms=parameters.get("ttl_ms"),
            )
        except Exception as exc:
            return ToolResult.error(
                f"发送消息失败: {exc}",
                error_type="send_message_failed",
                metadata={
                    "recipient_type": parameters.get("recipient_type"),
                    "recipient_id": parameters.get("recipient_id"),
                },
            )
        payload = {
            "recipientType": parameters["recipient_type"],
            "recipientId": parameters["recipient_id"],
            "senderId": sender_id,
            "ttlMs": parameters.get("ttl_ms"),
            "deliveryCount": len(deliveries),
            "deliveries": [message.to_dict() for message in deliveries],
        }
        return ToolResult.success(
            content=f"已发送 {len(deliveries)} 条消息",
            display_text=format_structured_display(
                f"已发送 {len(deliveries)} 条消息",
                payload,
            ),
            structured_data=payload,
            metadata=payload,
        )


def register_send_message_tool(
    registry: ToolRegistry,
    *,
    agent_runtime: AgentRuntimeManager,
    parent_agent: Any | None = None,
    expose_in_deferred: bool | None = True,
) -> SendMessageTool:
    tool = SendMessageTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = ["SendMessageTool", "register_send_message_tool"]
