"""Structured mailbox messaging for agent runtime collaboration."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from multiagent.default.agents import AgentRuntimeManager
from runtime import AgentRuntime

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.display_utils import format_structured_display


class SendMessageParams(BaseModel):
    to: str = Field(description='Recipient teammate name, agent ID, or "*" to broadcast')
    summary: str | None = Field(
        default=None,
        description="A 5-10 word summary shown as a preview",
    )
    message: str | dict[str, Any] = Field(description="Plain text or structured protocol message")


SEND_MESSAGE_PROMPT = """# SendMessage

Send a message to another agent.

```json
{"to": "researcher", "summary": "assign task 1", "message": "start on task #1"}
```

Use a teammate name for `to`, or `"*"` to broadcast to all teammates. Broadcast is linear in team size, so use it only when everyone genuinely needs the message.

Your plain text output is NOT visible to other agents. To communicate, you MUST call this tool. Messages from teammates are delivered automatically; do not check an inbox for normal delivery. Refer to teammates by name rather than opaque IDs when possible. When relaying a teammate message, do not quote it unnecessarily because it is already visible to the user.

For protocol messages such as shutdown or plan approval, pass the structured object as `message`. Do not send structured status messages for ordinary progress; use the shared task tools for task status."""


class SendMessageTool(Tool):
    def __init__(
        self,
        *,
        agent_runtime: AgentRuntimeManager,
        sender_name: str | None = None,
        runtime: AgentRuntime | None = None,
    ):
        self.agent_runtime = agent_runtime
        self.sender_name = sender_name
        self.runtime = runtime
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
        sender_id = self.sender_name
        metadata: dict[str, Any] = {}
        current_task_id = getattr(self.runtime, "current_task_id", None)
        if current_task_id and "taskId" not in metadata:
            metadata["taskId"] = current_task_id
        summary = str(parameters.get("summary") or "").strip()
        if summary:
            metadata["summary"] = summary
        message = parameters.get("message")
        content = message if isinstance(message, str) else json.dumps(message, ensure_ascii=False)
        target = str(parameters.get("to") or "").strip()
        context_metadata = dict(getattr(self.runtime, "metadata", {}) or {})
        if target == "*":
            recipient_type = "team"
            recipient_id = str(context_metadata.get("teamName") or "").strip()
            if not recipient_id:
                return ToolResult.error(
                    "Cannot broadcast outside a team context.",
                    error_type="send_message_failed",
                    metadata={"to": target},
                )
        else:
            recipient_type = "agent"
            recipient_id = target
        try:
            deliveries = self.agent_runtime.send_message(
                recipient_type=recipient_type,
                recipient_id=recipient_id,
                content=content,
                sender_id=sender_id,
                metadata=metadata,
            )
        except Exception as exc:
            return ToolResult.error(
                f"发送消息失败: {exc}",
                error_type="send_message_failed",
                metadata={
                    "to": target,
                },
            )
        payload = {
            "to": target,
            "recipientType": recipient_type,
            "recipientId": recipient_id,
            "senderId": sender_id,
            "summary": summary or None,
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
    sender_name: str | None = None,
    runtime: AgentRuntime | None = None,
    expose_in_deferred: bool | None = True,
) -> SendMessageTool:
    tool = SendMessageTool(
        agent_runtime=agent_runtime,
        sender_name=sender_name,
        runtime=runtime,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = ["SendMessageTool", "register_send_message_tool"]
