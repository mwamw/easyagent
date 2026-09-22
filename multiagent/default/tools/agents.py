"""Public tools for querying and controlling subagent runtime handles."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from multiagent.default.agents import AgentRuntimeManager
from task import TaskStatus

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.display_utils import format_structured_display


class AgentGetParams(BaseModel):
    agent_id: str = Field(description="Subagent ID to inspect")


class AgentListParams(BaseModel):
    status: Optional[str] = Field(default=None, description="Filter by status")
    team_id: Optional[str] = Field(default=None, description="Filter by team ID or name")
    current_task_id: Optional[str] = Field(default=None, description="Filter by current task ID")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum number of results")


class AgentWaitParams(BaseModel):
    agent_id: str = Field(description="Subagent ID to wait for")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="Wait timeout in milliseconds")


class AgentStopParams(BaseModel):
    agent_id: str = Field(description="Subagent ID to stop")
    reason: str = Field(default="", description="Reason for stopping")
    wait: bool = Field(default=False, description="Wait for the subagent to reach a terminal state")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="Wait timeout in milliseconds")


AGENT_GET_PROMPT = """Reads a subagent's full runtime handle.
- Use a known agent_id to inspect status, outputFile, executionContext, team/task bindings, and mailbox summary.
- This returns the current snapshot; it does not wait for completion. Use AgentWait to wait.
- Use AgentList to discover handles."""


AGENT_LIST_PROMPT = """Lists subagent handles in the current runtime.
- Filter by status, team_id, or current_task_id when many agents exist.
- Use returned agent_id values with AgentGet, AgentWait, or AgentStop.
- If the agent_id is already known, prefer AgentGet."""


AGENT_WAIT_PROMPT = """Waits for a specified subagent and returns its latest handle.
- Use after Agent(run_in_background=true) when its result is needed.
- timeout_ms limits the wait. A timeout is not an agent failure.
- Inspect status and outputFile to decide whether to wait again, collect results, or stop the agent."""


AGENT_STOP_PROMPT = """Sends a cooperative stop request to a background subagent.
- Use when its work is no longer needed or new constraints require stopping it.
- This is not a force-kill operation. The agent stops at a cooperative checkpoint.
- Use wait=true with timeout_ms to observe whether it actually stopped.
- Inspect status, stopReason, and outputFile; an accepted stop request does not imply a completed result."""


class _AgentRuntimeToolBase(Tool):
    def __init__(self, *, agent_runtime: AgentRuntimeManager, task_service: Any | None = None, **kwargs):
        self.agent_runtime = agent_runtime
        self.task_service = task_service
        super().__init__(**kwargs)

    def _sync_task_binding(self, handle: Any) -> None:
        task_service = self.task_service
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
    def __init__(self, *, agent_runtime: AgentRuntimeManager, task_service: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            task_service=task_service,
            name="AgentGet",
            description="Reads the structured runtime state of a subagent.",
            parameters=AgentGetParams,
            guidance="Inspect the subagent's current status, output file, and execution context.",
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
    def __init__(self, *, agent_runtime: AgentRuntimeManager, task_service: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            task_service=task_service,
            name="AgentList",
            description="Lists subagent handles in the current runtime.",
            parameters=AgentListParams,
            guidance="Discover agents, optionally filtered by status, team, or current_task_id.",
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
    def __init__(self, *, agent_runtime: AgentRuntimeManager, task_service: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            task_service=task_service,
            name="AgentWait",
            description="Waits for a subagent and returns its latest observable state.",
            parameters=AgentWaitParams,
            guidance="Wait for a background subagent when its result is needed.",
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
    def __init__(self, *, agent_runtime: AgentRuntimeManager, task_service: Any | None = None):
        super().__init__(
            agent_runtime=agent_runtime,
            task_service=task_service,
            name="AgentStop",
            description="Requests cooperative shutdown of a background subagent.",
            parameters=AgentStopParams,
            guidance="This requests cooperative stopping, with an optional wait for a terminal state.",
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
    task_service: Any | None = None,
    expose_in_deferred: bool | None = True,
) -> tuple[AgentGetTool, AgentListTool, AgentWaitTool, AgentStopTool]:
    get_tool = AgentGetTool(agent_runtime=agent_runtime, task_service=task_service)
    list_tool = AgentListTool(agent_runtime=agent_runtime, task_service=task_service)
    wait_tool = AgentWaitTool(agent_runtime=agent_runtime, task_service=task_service)
    stop_tool = AgentStopTool(agent_runtime=agent_runtime, task_service=task_service)
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
