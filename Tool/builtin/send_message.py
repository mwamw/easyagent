"""Structured mailbox messaging for agent runtime collaboration."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from runtime import AgentRuntimeManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


class SendMessageParams(BaseModel):
    recipient_type: Literal["agent", "team", "task"] = Field(description="消息接收方类型")
    recipient_id: str = Field(description="接收方 ID，team 类型时也支持团队名称")
    content: str = Field(description="要发送的结构化文本消息")
    sender_id: Optional[str] = Field(default=None, description="发送方标识，不填则使用当前 agent 名称")
    ttl_ms: Optional[int] = Field(default=None, ge=0, description="消息存活时间，毫秒；超时后会标记为 expired")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


SEND_MESSAGE_PROMPT = """向 agent、team 或 task 作用域发送结构化运行时消息。

何时使用：
- 当 manager 需要给某个子 agent 补充新约束、新上下文、下一步动作或优先级变更。
- 当你需要把同一条消息广播给整个团队，或广播给绑定在同一 `task_id` 下的一组 agent。
- 当后台子 agent 已经启动，而你希望在后续轮次动态调整它的行为。

接收方语义：
- `recipient_type=\"agent\"`：只发给单个 agent。
- `recipient_type=\"team\"`：广播给团队当前所有成员。
- `recipient_type=\"task\"`：广播给当前绑定到某个结构化 task 的所有 agent。

重要限制：
- `SendMessage` 只保证“消息送达到 mailbox”，不保证“消息已经被对方消费”。
- 消息送达后，接收方需要通过自动注入的 mailbox prompt 或 `MailboxRead` 看到它，再用 `MailboxAck` 确认消费。
- 如果你要求对方必须在本轮之后改变行为，应在后续用 `AgentGet`、`AgentWait`、`MailboxRead` 或 mailbox 状态检查来验证。

最佳实践：
- `content` 要具体，写明新的要求、边界、优先级或禁止事项。
- 需要追踪上下文时，把 `taskId`、来源阶段、约束标签等信息放进 `metadata`。
- 只希望消息在短时间内有效时，设置 `ttl_ms`，避免过期消息长期污染协作状态。"""


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
            guidance=(
                "适合在多 agent 协作中广播约束、补充上下文或指定下一步动作。"
                " 发送成功只代表消息已写入 mailbox，不代表对方已经消费。"
            ),
            prompt=SEND_MESSAGE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
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
) -> SendMessageTool:
    tool = SendMessageTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    registry.register_tool(tool)
    return tool


__all__ = ["SendMessageTool", "register_send_message_tool"]
