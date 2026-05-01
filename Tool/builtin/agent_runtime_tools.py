"""Public tools for querying and controlling subagent runtime handles."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from runtime import AgentRuntimeManager
from task import TaskStatus

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


class AgentGetParams(BaseModel):
    agent_id: str = Field(description="要查询的子 agent ID")


class AgentListParams(BaseModel):
    status: Optional[str] = Field(default=None, description="按状态过滤")
    team_id: Optional[str] = Field(default=None, description="按团队 ID 或团队名称过滤")
    current_task_id: Optional[str] = Field(default=None, description="按当前任务 ID 过滤")
    limit: int = Field(default=100, ge=1, le=500, description="返回数量上限")


class AgentWaitParams(BaseModel):
    agent_id: str = Field(description="要等待的子 agent ID")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="等待超时时间，毫秒")


class AgentStopParams(BaseModel):
    agent_id: str = Field(description="要停止的子 agent ID")
    reason: str = Field(default="", description="停止原因")
    wait: bool = Field(default=False, description="是否等待子 agent 进入终态")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="等待超时时间，毫秒")


AGENT_GET_PROMPT = """读取单个子 agent 的完整运行时句柄。

适合：
- 你已经知道 `agent_id`，想查看它的状态、outputFile、executionContext、team/task 绑定和 mailbox 概况。
- 你需要确认后台子 agent 当前是否还在运行、是否已报错、是否已有输出。

不要用它做什么：
- 不要用 `AgentGet` 代替 `AgentWait` 等待完成；它只是读取当前状态快照。
- 如果你想看所有 handles 的整体分布，优先用 `AgentList`。"""


AGENT_LIST_PROMPT = """列出当前 runtime 中的子 agent handles，并支持按状态、team、task 过滤。

适合：
- 你需要发现当前有哪些后台/前台子 agent。
- 你需要查看某个 team 或 task 下面有哪些 agent 正在运行。
- 你需要在多个 handles 之间挑出目标，再配合 `AgentGet` / `AgentWait` / `AgentStop` 使用。

最佳实践：
- agent 数量较多时，先加 `status`、`team_id` 或 `current_task_id` 过滤。
- 如果已经知道具体 `agent_id`，优先用 `AgentGet`。"""


AGENT_WAIT_PROMPT = """阻塞等待指定子 agent 到达终态或当前可观察状态。

适合：
- `Agent(run_in_background=true)` 之后等待结果落定，再继续汇总。
- 你需要在继续执行前确认某个子 agent 是否已完成、报错或停止。

重要语义：
- 超时不代表失败，只表示在 `timeout_ms` 内没有进入你希望观察到的状态。
- 超时后应继续看返回的 handle、status、outputFile，并决定是继续等待还是中断。
- `AgentWait` 返回的是最新 handle，不只是一个布尔结果。"""


AGENT_STOP_PROMPT = """向后台子 agent 发送协作停止请求，并可选择等待它进入终态。

适合：
- 当前任务已不再需要该子 agent 继续运行。
- 你发现子 agent 跑偏、跑得太久，或收到新的高优先级约束，需要中止它。

重要语义：
- 这不是强杀进程语义，而是协作停止协议；默认 BasicAgent 子 agent 会收到停止请求并尽量优雅退出。
- `wait=true` 时通常应配合 `timeout_ms`，确保你能确认它是否真正进入终态。
- 停止请求成功不代表结果可用；停止后应检查返回 handle 中的 `status`、`stopReason` 和 `outputFile`。"""


class _AgentRuntimeToolBase(Tool):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None, **kwargs):
        self.agent_runtime = agent_runtime
        self.parent_agent = parent_agent
        super().__init__(**kwargs)

    def _sync_task_binding(self, handle: Any) -> None:
        task_service = getattr(self.parent_agent, "task_service", None)
        task_id = getattr(handle, "metadata", {}).get("task_id")
        if task_service is None or not task_id:
            return
        status_map = {
            "completed": TaskStatus.COMPLETED,
            "error": TaskStatus.BLOCKED,
            "stopped": TaskStatus.CANCELLED,
            "interrupted": TaskStatus.BLOCKED,
            "stop_requested": TaskStatus.BLOCKED,
            "async_launched": TaskStatus.IN_PROGRESS,
            "running": TaskStatus.IN_PROGRESS,
        }
        runtime_metadata = dict((handle.metadata or {}).get("runtime") or {})
        runtime_metadata.update(
            {
                "agentId": handle.agent_id,
                "teamId": getattr(handle, "team_id", None),
                "teamName": getattr(handle, "team_name", None),
                "outputFile": handle.output_file,
                "status": handle.status,
            }
        )
        try:
            task_service.update_task(
                task_id,
                owner=handle.agent_id,
                status=status_map.get(handle.status, TaskStatus.IN_PROGRESS),
                metadata={"runtime": runtime_metadata},
            )
        except Exception:
            return


class AgentGetTool(_AgentRuntimeToolBase):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            parent_agent=parent_agent,
            name="AgentGet",
            description="读取单个子 agent 的结构化运行时状态。",
            parameters=AgentGetParams,
            guidance="适合在多 agent 协作中查看指定子 agent 的当前状态、输出文件和 execution context。",
            prompt=AGENT_GET_PROMPT,
            read_only=True,
            supports_parallel=True,
            source="builtin",
            tags=["agent", "runtime", "query"],
            side_effect_level="none",
            resource_scope=["runtime", "task"],
        )

    def run(self, parameters: dict) -> ToolResult:
        agent_id = str(parameters.get("agent_id") or "").strip()
        try:
            handle = self.agent_runtime.get_handle(agent_id)
        except Exception as exc:
            return ToolResult.error(
                f"查询子 agent 失败: {exc}",
                error_type="agent_not_found",
                metadata={"agent_id": agent_id},
            )
        self._sync_task_binding(handle)
        payload = handle.to_tool_payload()
        return ToolResult.success(
            content=f"已读取子 agent {handle.agent_id} 的运行时状态",
            display_text=format_structured_display(
                f"已读取子 agent {handle.agent_id} 的运行时状态",
                payload,
                result_text=getattr(handle, "content", ""),
            ),
            structured_data=payload,
            metadata=payload,
        )


class AgentListTool(_AgentRuntimeToolBase):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            parent_agent=parent_agent,
            name="AgentList",
            description="列出当前 runtime 中的子 agent handles。",
            parameters=AgentListParams,
            guidance="适合查看所有子 agent，或按状态、team、current_task_id 过滤。",
            prompt=AGENT_LIST_PROMPT,
            read_only=True,
            supports_parallel=True,
            source="builtin",
            tags=["agent", "runtime", "query"],
            side_effect_level="none",
            resource_scope=["runtime", "task"],
        )

    def run(self, parameters: dict) -> ToolResult:
        team_filter = parameters.get("team_id")
        handles = self.agent_runtime.list_handles(
            status=parameters.get("status"),
            current_task_id=parameters.get("current_task_id"),
        )
        if team_filter:
            handles = [
                handle for handle in handles
                if handle.team_id == team_filter or handle.team_name == team_filter
            ]
        limit = parameters.get("limit")
        if limit is not None:
            handles = handles[: max(int(limit), 0)]
        for handle in handles:
            self._sync_task_binding(handle)
        payload = {
            "count": len(handles),
            "filters": {
                "status": parameters.get("status"),
                "teamId": team_filter,
                "currentTaskId": parameters.get("current_task_id"),
                "limit": parameters.get("limit"),
            },
            "agents": [handle.to_tool_payload() for handle in handles],
        }
        return ToolResult.success(
            content=f"已列出 {len(handles)} 个子 agent",
            display_text=format_structured_display(
                f"已列出 {len(handles)} 个子 agent",
                payload,
            ),
            structured_data=payload,
            metadata=payload,
        )


class AgentWaitTool(_AgentRuntimeToolBase):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            parent_agent=parent_agent,
            name="AgentWait",
            description="阻塞等待指定子 agent 到达当前可观察状态或终态。",
            parameters=AgentWaitParams,
            guidance="适合在后台子 agent 运行后等待它完成，再读取结构化 handle。",
            prompt=AGENT_WAIT_PROMPT,
            read_only=True,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "runtime", "wait"],
            side_effect_level="none",
            resource_scope=["runtime", "task"],
        )

    def run(self, parameters: dict) -> ToolResult:
        agent_id = str(parameters.get("agent_id") or "").strip()
        timeout_ms = parameters.get("timeout_ms")
        timed_out = False
        try:
            handle = self.agent_runtime.wait(agent_id, timeout_ms=timeout_ms)
        except TimeoutError:
            timed_out = True
            handle = self.agent_runtime.get_handle(agent_id)
        except Exception as exc:
            return ToolResult.error(
                f"等待子 agent 失败: {exc}",
                error_type="agent_wait_failed",
                metadata={"agent_id": agent_id, "timeout_ms": timeout_ms},
            )
        self._sync_task_binding(handle)
        payload = handle.to_tool_payload()
        payload["timedOut"] = timed_out
        payload["timeoutMs"] = timeout_ms
        message = (
            f"等待子 agent 超时: {handle.agent_id}"
            if timed_out
            else f"子 agent 已到达可观察状态: {handle.agent_id}"
        )
        return ToolResult.success(
            content=message,
            display_text=format_structured_display(
                message,
                payload,
                result_text=getattr(handle, "content", ""),
            ),
            structured_data=payload,
            metadata=payload,
        )


class AgentStopTool(_AgentRuntimeToolBase):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, parent_agent: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            parent_agent=parent_agent,
            name="AgentStop",
            description="请求停止一个后台子 agent，可选择等待其进入终态。",
            parameters=AgentStopParams,
            guidance="对默认 BasicAgent 子 agent，这是协作停止协议：先发停止请求，再按需等待终态。",
            prompt=AGENT_STOP_PROMPT,
            read_only=False,
            destructive=True,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "runtime", "control"],
            risk_categories=["side_effect"],
            side_effect_level="medium",
            resource_scope=["runtime", "task"],
        )

    def run(self, parameters: dict) -> ToolResult:
        agent_id = str(parameters.get("agent_id") or "").strip()
        wait = bool(parameters.get("wait", False))
        timeout_ms = parameters.get("timeout_ms")
        reason = str(parameters.get("reason") or "").strip()
        timed_out = False
        try:
            handle = self.agent_runtime.stop(
                agent_id,
                reason=reason,
                wait=wait,
                timeout_ms=timeout_ms,
            )
        except TimeoutError:
            timed_out = True
            handle = self.agent_runtime.get_handle(agent_id)
        except Exception as exc:
            return ToolResult.error(
                f"停止子 agent 失败: {exc}",
                error_type="agent_stop_failed",
                metadata={"agent_id": agent_id},
            )
        self._sync_task_binding(handle)
        payload = handle.to_tool_payload()
        payload["timedOut"] = timed_out
        payload["timeoutMs"] = timeout_ms
        payload["waited"] = wait
        payload["requestedReason"] = reason
        message = (
            f"已发送停止请求，但等待超时: {handle.agent_id}"
            if timed_out
            else f"已更新子 agent 停止状态: {handle.agent_id}"
        )
        return ToolResult.success(
            content=message,
            display_text=format_structured_display(
                message,
                payload,
                result_text=getattr(handle, "content", ""),
            ),
            structured_data=payload,
            metadata=payload,
        )


def register_agent_runtime_tools(
    registry: ToolRegistry,
    *,
    agent_runtime: AgentRuntimeManager,
    parent_agent: Any | None = None,
    expose_in_deferred: bool | None = True,
) -> tuple[AgentGetTool, AgentListTool, AgentWaitTool, AgentStopTool]:
    get_tool = AgentGetTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    list_tool = AgentListTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    wait_tool = AgentWaitTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    stop_tool = AgentStopTool(agent_runtime=agent_runtime, parent_agent=parent_agent)
    registry.register_tool(get_tool, expose_in_deferred=expose_in_deferred)
    registry.register_tool(list_tool, expose_in_deferred=expose_in_deferred)
    registry.register_tool(wait_tool, expose_in_deferred=expose_in_deferred)
    registry.register_tool(stop_tool, expose_in_deferred=expose_in_deferred)
    return get_tool, list_tool, wait_tool, stop_tool


__all__ = [
    "AgentGetTool",
    "AgentListTool",
    "AgentStopTool",
    "AgentWaitTool",
    "register_agent_runtime_tools",
]
