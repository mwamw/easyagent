"""Structured mailbox messaging for agent runtime collaboration."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from runtime import AgentRuntimeManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry


class SendMessageParams(BaseModel):
    recipient_type: Literal["agent", "team"] = Field(description="消息接收方类型")
    recipient_id: str = Field(description="接收方 ID，team 类型时也支持团队名称")
    content: str = Field(description="要发送的结构化文本消息")
    sender_id: Optional[str] = Field(default=None, description="发送方标识，不填则使用当前 agent 名称")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


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
            description="向指定 agent 或团队发送结构化消息，写入运行时 mailbox。",
            parameters=SendMessageParams,
            guidance="适合在多 agent 协作中广播约束、补充上下文或指定下一步动作。",
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
        )

    def run(self, parameters: dict) -> ToolResult:
        sender_id = parameters.get("sender_id") or getattr(self.parent_agent, "name", None)
        try:
            deliveries = self.agent_runtime.send_message(
                recipient_type=parameters["recipient_type"],
                recipient_id=parameters["recipient_id"],
                content=parameters["content"],
                sender_id=sender_id,
                metadata=parameters.get("metadata") or {},
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
            "deliveryCount": len(deliveries),
            "deliveries": [message.to_dict() for message in deliveries],
        }
        return ToolResult.success(
            content=f"已发送 {len(deliveries)} 条消息",
            structured_data=payload,
            metadata=payload,
        )


def register_send_message_tool(
    registry: ToolRegistry,
    *,
    agent_runtime: AgentRuntimeManager,
    parent_agent: Any | None = None,
) -> SendMessageTool:
    tool = SendMessageTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    registry.register_tool(tool)
    return tool


__all__ = ["SendMessageTool", "register_send_message_tool"]
