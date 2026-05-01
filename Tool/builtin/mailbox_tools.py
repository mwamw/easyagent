"""Mailbox read/ack tools for multi-agent collaboration."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from runtime import AgentRuntimeManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


MAILBOX_READ_PROMPT = """读取某个 agent mailbox 中当前可见的消息，并可把 `queued` 消息标记为 `delivered`。

何时使用：
- 当你需要查看某个子 agent 或当前 agent 最近收到的运行时消息。
- 当你需要确认 team 广播、task 广播或点对点消息是否已经送达。
- 当你需要读取自动注入到系统提示之外的完整结构化 mailbox payload。

关键语义：
- `queued` 表示消息已送达 mailbox，但该 agent 还未显式读取。
- `delivered` 表示消息已经被读取或自动注入到 prompt 中，但还没有被确认消费。
- `consumed` 表示消息已经被 `MailboxAck` 确认处理。
- `expired` 表示消息超过 TTL，不应再作为当前执行输入。

使用建议：
- 对当前 agent，通常不需要手填 `agent_id`；工具会默认读取自己的 mailbox。
- 如果你只是想确认系统提示里已经自动注入了哪些协作消息，也可以直接调用本工具。
- 读取后如果你已经把消息纳入执行，应再调用 `MailboxAck`。
- 如果只是排查协作状态，可设置 `include_consumed` 或 `include_expired` 查看完整历史。"""


MAILBOX_ACK_PROMPT = """把 mailbox 消息标记为 `consumed`，表示当前 agent 已经处理了这些运行时输入。

何时使用：
- 当你已经阅读并执行了某条 manager / team / task 消息。
- 当你希望从“待处理消息”中移除已处理项，避免下一轮继续重复看到它们。
- 当你在复杂协作流中需要明确区分“消息已送达”和“消息已消费”。

关键语义：
- `MailboxAck` 不会删除消息，只会把状态改成 `consumed` 并记录 ack 行为。
- 如果要一次性确认当前 agent 的所有待处理消息，可使用 `ack_all=true`。
- 如果只想确认部分消息，应精确传入 `message_ids`。

注意：
- 不要在真正阅读之前先 ack；这会让后续协作状态变得不准确。
- 如果消息仍然需要保留为未处理输入，就不要 ack。"""


class MailboxReadParams(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="[重要]如果你是想查收自己收到的协作消息，请务必留空不填！切勿填入文件名或任务名。仅仅在你是 manager 替其他 agent 查询时才填目标 agentId。")
    limit: int = Field(default=100, ge=1, le=500, description="返回消息数量上限")
    include_consumed: bool = Field(default=False, description="是否包含已消费消息")
    include_expired: bool = Field(default=False, description="是否包含已过期消息")
    mark_delivered: bool = Field(default=True, description="是否把 queued 消息标记为 delivered")


class MailboxAckParams(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="[重要]如果是确认你自己收到的消息，请务必留空不填！")
    message_ids: list[str] = Field(default_factory=list, description="要确认消费的消息 ID 列表")
    ack_all: bool = Field(default=False, description="是否确认当前 agent 所有未消费消息")


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
            description="读取 agent runtime mailbox，并可把 queued 消息标记为 delivered。",
            parameters=MailboxReadParams,
            guidance=(
                "适合在多 agent 协作中读取当前 agent 或指定 agent 的运行时消息。"
                " 如果消息已经自动注入 prompt，本工具可用于再次查看完整结构化载荷。"
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
            description="确认 mailbox 消息已被当前 agent 处理，把状态更新为 consumed。",
            parameters=MailboxAckParams,
            guidance=(
                "适合在多 agent 协作中显式确认消息已被处理。"
                " 这会把“已送达”和“已消费”区分开，便于 manager 跟踪协作进度。"
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
