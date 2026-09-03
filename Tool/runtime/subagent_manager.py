"""Runtime manager for Claude-style sub-agent execution."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import (
    CancelledError as FutureCancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

from core.Exception import AgentStopRequested


@dataclass(slots=True)
class SubagentRequest:
    description: str
    prompt: str
    agent_type: Optional[str] = None
    model: Optional[str] = None
    name: Optional[str] = None
    team_name: Optional[str] = None
    mode: Optional[str] = None
    isolation: Optional[str] = None
    workspace_root: Optional[str] = None
    allowed_roots: tuple[str, ...] = ()
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "prompt": self.prompt,
            "agentType": self.agent_type,
            "model": self.model,
            "name": self.name,
            "teamName": self.team_name,
            "mode": self.mode,
            "isolation": self.isolation,
            "workspaceRoot": self.workspace_root,
            "allowedRoots": list(self.allowed_roots),
            "worktreePath": self.worktree_path,
            "worktreeBranch": self.worktree_branch,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SubagentRequest":
        data = dict(payload or {})
        return cls(
            description=str(data.get("description") or ""),
            prompt=str(data.get("prompt") or ""),
            agent_type=data.get("agentType"),
            model=data.get("model"),
            name=data.get("name"),
            team_name=data.get("teamName"),
            mode=data.get("mode"),
            isolation=data.get("isolation"),
            workspace_root=data.get("workspaceRoot"),
            allowed_roots=tuple(str(item) for item in list(data.get("allowedRoots") or [])),
            worktree_path=data.get("worktreePath"),
            worktree_branch=data.get("worktreeBranch"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class SubagentSnapshot:
    agent_id: str
    status: str
    description: str
    prompt: str
    output_file: str
    agent_type: Optional[str] = None
    name: Optional[str] = None
    team_name: Optional[str] = None
    mode: Optional[str] = None
    isolation: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    started_at: float = 0.0
    finished_at: Optional[float] = None
    content: str = ""
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    total_duration_ms: int = 0
    total_tool_use_count: int = 0
    total_tokens: int = 0
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "status": self.status,
            "description": self.description,
            "prompt": self.prompt,
            "outputFile": self.output_file,
            "agentType": self.agent_type,
            "name": self.name,
            "teamName": self.team_name,
            "mode": self.mode,
            "isolation": self.isolation,
            "worktreePath": self.worktree_path,
            "worktreeBranch": self.worktree_branch,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "content": self.content,
            "error": self.error,
            "stopReason": self.stop_reason,
            "totalDurationMs": self.total_duration_ms,
            "totalToolUseCount": self.total_tool_use_count,
            "totalTokens": self.total_tokens,
            "usage": dict(self.usage),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SubagentSnapshot":
        data = dict(payload or {})
        return cls(
            agent_id=str(data.get("agentId") or ""),
            status=str(data.get("status") or "completed"),
            description=str(data.get("description") or ""),
            prompt=str(data.get("prompt") or ""),
            output_file=str(data.get("outputFile") or ""),
            agent_type=data.get("agentType"),
            name=data.get("name"),
            team_name=data.get("teamName"),
            mode=data.get("mode"),
            isolation=data.get("isolation"),
            worktree_path=data.get("worktreePath"),
            worktree_branch=data.get("worktreeBranch"),
            started_at=float(data.get("startedAt") or 0.0),
            finished_at=float(data["finishedAt"]) if data.get("finishedAt") is not None else None,
            content=str(data.get("content") or ""),
            error=data.get("error"),
            stop_reason=data.get("stopReason"),
            total_duration_ms=int(data.get("totalDurationMs") or 0),
            total_tool_use_count=int(data.get("totalToolUseCount") or 0),
            total_tokens=int(data.get("totalTokens") or 0),
            usage=dict(data.get("usage") or {}),
        )


class SubagentManager:
    def __init__(
        self,
        *,
        agent_factory: Callable[[SubagentRequest], Any],
        storage_dir: Optional[str] = None,
        max_background_tasks: int = 4,
    ):
        self.agent_factory = agent_factory
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".easyagent-agents"))
        self.max_background_tasks = max(1, int(max_background_tasks))
        self._executor = ThreadPoolExecutor(max_workers=self.max_background_tasks, thread_name_prefix="easyagent-subagent")
        self._lock = threading.RLock()
        self._snapshots: dict[str, SubagentSnapshot] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._agents: dict[str, Any] = {}
        self.last_restore_report: Optional[dict[str, Any]] = None
        self.last_close_report: Optional[dict[str, Any]] = None
        os.makedirs(self.storage_dir, exist_ok=True)

    def _build_output_file(self, agent_id: str) -> str:
        return os.path.join(self.storage_dir, f"{agent_id}.md")

    def _write_output_file(self, path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_snapshot_output(
        self,
        snapshot: SubagentSnapshot,
        request: SubagentRequest,
        *,
        status: str,
        section_title: str,
        body: str,
        finished_at: float,
    ) -> None:
        self._write_output_file(
            snapshot.output_file,
            (
                f"# {request.description or snapshot.agent_id}\n\n"
                f"状态: {status}\n"
                f"开始时间: {snapshot.started_at}\n"
                f"结束时间: {finished_at}\n\n"
                f"## Prompt\n{request.prompt}\n\n"
                f"## {section_title}\n{body}\n"
            ),
        )

    def _create_snapshot(self, request: SubagentRequest, *, status: str) -> SubagentSnapshot:
        agent_id = f"agent_{uuid4().hex[:12]}"
        output_file = self._build_output_file(agent_id)
        snapshot = SubagentSnapshot(
            agent_id=agent_id,
            status=status,
            description=request.description,
            prompt=request.prompt,
            output_file=output_file,
            agent_type=request.agent_type,
            name=request.name,
            team_name=request.team_name,
            mode=request.mode,
            isolation=request.isolation,
            worktree_path=request.worktree_path,
            worktree_branch=request.worktree_branch,
            started_at=time.time(),
        )
        with self._lock:
            self._snapshots[agent_id] = snapshot
        return snapshot

    def _count_tool_calls(self, agent: Any) -> int:
        try:
            trace_history = list(getattr(agent, "get_trace_history")())
        except Exception:
            trace_history = list(getattr(agent, "trace_history", []) or [])
        return sum(
            1
            for event in trace_history
            if isinstance(event, dict) and event.get("type") == "tool.invoke.started"
        )

    def _build_usage(self, agent: Any) -> tuple[dict[str, Any], int]:
        usage: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "server_tool_use": {
                "web_search_requests": 0,
                "web_fetch_requests": 0,
            },
            "service_tier": None,
            "cache_creation": None,
        }
        try:
            context_usage = agent.get_context_usage()
        except Exception:
            context_usage = {}

        input_tokens = 0
        if isinstance(context_usage, dict):
            input_tokens = int(context_usage.get("estimatedRequestTokens") or 0)

        try:
            trace_history = list(getattr(agent, "get_trace_history")())
        except Exception:
            trace_history = list(getattr(agent, "trace_history", []) or [])

        llm_usage_seen = False
        for event in trace_history:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            data = dict(event.get("data") or {})
            if event_type == "llm.invoke.completed":
                metrics = dict(data.get("usage") or {})
                llm_usage_seen = llm_usage_seen or any(
                    metrics.get(key) is not None
                    for key in ("inputTokens", "outputTokens", "totalTokens")
                )
                usage["input_tokens"] += int(metrics.get("inputTokens") or 0)
                usage["output_tokens"] += int(metrics.get("outputTokens") or 0)
                cache_creation = int(metrics.get("cacheCreationTokens") or 0)
                cache_read = int(metrics.get("cacheReadTokens") or metrics.get("cachedInputTokens") or 0)
                usage["cache_creation_input_tokens"] = int(usage["cache_creation_input_tokens"] or 0) + cache_creation
                usage["cache_read_input_tokens"] = int(usage["cache_read_input_tokens"] or 0) + cache_read
                continue
            if event_type != "tool.invoke.started":
                continue
            tool_name = str(data.get("tool_name") or "")
            if tool_name == "WebSearch":
                usage["server_tool_use"]["web_search_requests"] += 1
            elif tool_name == "WebFetch":
                usage["server_tool_use"]["web_fetch_requests"] += 1

        if not llm_usage_seen:
            usage["input_tokens"] = input_tokens
        if not usage["cache_creation_input_tokens"]:
            usage["cache_creation_input_tokens"] = None
        if not usage["cache_read_input_tokens"]:
            usage["cache_read_input_tokens"] = None

        total_tokens = int(usage["input_tokens"] or 0) + int(usage["output_tokens"] or 0)
        return usage, total_tokens

    @staticmethod
    def _mark_terminal_snapshot(
        snapshot: SubagentSnapshot,
        *,
        status: str,
        finished_at: float,
        content: str = "",
        error: Optional[str] = None,
        stop_reason: Optional[str] = None,
        total_tool_use_count: int = 0,
        total_tokens: int = 0,
        usage: Optional[dict[str, Any]] = None,
    ) -> None:
        snapshot.status = status
        snapshot.content = content
        snapshot.error = error
        snapshot.stop_reason = stop_reason
        snapshot.finished_at = finished_at
        snapshot.total_duration_ms = int((finished_at - snapshot.started_at) * 1000)
        snapshot.total_tool_use_count = total_tool_use_count
        snapshot.total_tokens = total_tokens
        snapshot.usage = dict(usage or {})

    def _execute(
        self,
        snapshot: SubagentSnapshot,
        request: SubagentRequest,
        *,
        agent_factory: Optional[Callable[[SubagentRequest], Any]] = None,
    ) -> SubagentSnapshot:
        try:
            with self._lock:
                if snapshot.status == "async_launched":
                    snapshot.status = "running"
            request.metadata = {
                **dict(request.metadata or {}),
                "agent_id": snapshot.agent_id,
                "output_file": snapshot.output_file,
            }
            agent = (agent_factory or self.agent_factory)(request)
            with self._lock:
                self._agents[snapshot.agent_id] = agent
            result = agent.invoke(request.prompt)
            usage, total_tokens = self._build_usage(agent)
            total_tool_use_count = self._count_tool_calls(agent)
            content = str(result)
            finished_at = time.time()
            self._write_snapshot_output(
                snapshot,
                request,
                status="completed",
                section_title="Result",
                body=content,
                finished_at=finished_at,
            )
            with self._lock:
                self._mark_terminal_snapshot(
                    snapshot,
                    status="completed",
                    content=content,
                    finished_at=finished_at,
                    total_tool_use_count=total_tool_use_count,
                    total_tokens=total_tokens,
                    usage=usage,
                )
            return snapshot
        except AgentStopRequested as exc:
            finished_at = time.time()
            stop_reason = str(exc)
            usage = {}
            total_tokens = 0
            total_tool_use_count = 0
            try:
                agent = self._agents.get(snapshot.agent_id)
                if agent is not None:
                    usage, total_tokens = self._build_usage(agent)
                    total_tool_use_count = self._count_tool_calls(agent)
            except Exception:
                usage = {}
                total_tokens = 0
                total_tool_use_count = 0
            self._write_snapshot_output(
                snapshot,
                request,
                status="stopped",
                section_title="Stop",
                body=stop_reason,
                finished_at=finished_at,
            )
            with self._lock:
                self._mark_terminal_snapshot(
                    snapshot,
                    status="stopped",
                    finished_at=finished_at,
                    stop_reason=stop_reason,
                    total_tool_use_count=total_tool_use_count,
                    total_tokens=total_tokens,
                    usage=usage,
                )
            return snapshot
        except Exception as exc:
            finished_at = time.time()
            error_text = str(exc)
            self._write_snapshot_output(
                snapshot,
                request,
                status="error",
                section_title="Error",
                body=error_text,
                finished_at=finished_at,
            )
            with self._lock:
                self._mark_terminal_snapshot(
                    snapshot,
                    status="error",
                    finished_at=finished_at,
                    error=error_text,
                )
            return snapshot
        finally:
            with self._lock:
                self._agents.pop(snapshot.agent_id, None)

    def run(
        self,
        request: SubagentRequest,
        *,
        agent_factory: Optional[Callable[[SubagentRequest], Any]] = None,
        on_created: Optional[Callable[[SubagentSnapshot, SubagentRequest], None]] = None,
    ) -> SubagentSnapshot:
        snapshot = self._create_snapshot(request, status="running")
        request.metadata = {
            **dict(request.metadata or {}),
            "agent_id": snapshot.agent_id,
            "output_file": snapshot.output_file,
        }
        self._write_output_file(
            snapshot.output_file,
            f"# {request.description or snapshot.agent_id}\n\n状态: running\n\n## Prompt\n{request.prompt}\n",
        )
        if on_created is not None:
            on_created(snapshot, request)
        return self._execute(snapshot, request, agent_factory=agent_factory)

    def launch_background(
        self,
        request: SubagentRequest,
        *,
        agent_factory: Optional[Callable[[SubagentRequest], Any]] = None,
        on_created: Optional[Callable[[SubagentSnapshot, SubagentRequest], None]] = None,
    ) -> SubagentSnapshot:
        snapshot = self._create_snapshot(request, status="async_launched")
        request.metadata = {
            **dict(request.metadata or {}),
            "agent_id": snapshot.agent_id,
            "output_file": snapshot.output_file,
        }
        self._write_output_file(
            snapshot.output_file,
            (
                f"# {request.description or snapshot.agent_id}\n\n"
                "状态: async_launched\n\n"
                "## Prompt\n"
                f"{request.prompt}\n"
            ),
        )
        if on_created is not None:
            on_created(snapshot, request)
        future = self._executor.submit(
            self._execute,
            snapshot,
            request,
            agent_factory=agent_factory,
        )
        with self._lock:
            self._futures[snapshot.agent_id] = future
        return snapshot

    def get_snapshot(self, agent_id: str) -> SubagentSnapshot:
        with self._lock:
            snapshot = self._snapshots.get(agent_id)
        if snapshot is None:
            raise KeyError(f"子 agent 不存在: {agent_id}")

        future = self._futures.get(agent_id)
        if future is not None and future.done():
            try:
                future.result()
            except FutureCancelledError:
                pass
        return snapshot

    def list_snapshots(self) -> list[SubagentSnapshot]:
        with self._lock:
            agent_ids = list(self._snapshots.keys())
        return [self.get_snapshot(agent_id) for agent_id in agent_ids]

    def wait(self, agent_id: str, *, timeout_ms: Optional[int] = None) -> SubagentSnapshot:
        snapshot = self.get_snapshot(agent_id)
        with self._lock:
            future = self._futures.get(agent_id)
        if future is None:
            return snapshot
        try:
            timeout_s = None if timeout_ms is None else max(float(timeout_ms) / 1000.0, 0.0)
            future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            raise TimeoutError(f"等待子 agent 超时: {agent_id}") from exc
        except FutureCancelledError:
            return self.get_snapshot(agent_id)
        return self.get_snapshot(agent_id)

    def stop(
        self,
        agent_id: str,
        *,
        reason: str = "",
        wait: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> SubagentSnapshot:
        snapshot = self.get_snapshot(agent_id)
        if snapshot.status in {"completed", "error", "stopped", "cancelled", "interrupted"}:
            return snapshot

        stop_reason = str(reason or "").strip() or "外部请求停止该子 agent。"

        with self._lock:
            future = self._futures.get(agent_id)
            agent = self._agents.get(agent_id)

        if future is not None and future.cancel():
            finished_at = time.time()
            self._write_snapshot_output(
                snapshot,
                SubagentRequest(
                    description=snapshot.description,
                    prompt=snapshot.prompt,
                    agent_type=snapshot.agent_type,
                    name=snapshot.name,
                    team_name=snapshot.team_name,
                    mode=snapshot.mode,
                    isolation=snapshot.isolation,
                    worktree_path=snapshot.worktree_path,
                    worktree_branch=snapshot.worktree_branch,
                ),
                status="stopped",
                section_title="Stop",
                body=stop_reason,
                finished_at=finished_at,
            )
            with self._lock:
                self._mark_terminal_snapshot(
                    snapshot,
                    status="stopped",
                    finished_at=finished_at,
                    stop_reason=stop_reason,
                )
                self._futures.pop(agent_id, None)
            return snapshot

        request_stop = getattr(agent, "request_stop", None)
        if not callable(request_stop):
            raise RuntimeError(f"子 agent 当前不支持协作停止: {agent_id}")

        request_stop(stop_reason)
        with self._lock:
            snapshot.status = "stop_requested"
            snapshot.stop_reason = stop_reason

        if wait:
            return self.wait(agent_id, timeout_ms=timeout_ms)
        return snapshot

    def delete(self, agent_id: str, *, remove_output_file: bool = False) -> SubagentSnapshot:
        snapshot = self.get_snapshot(agent_id)
        with self._lock:
            self._snapshots.pop(agent_id, None)
            self._futures.pop(agent_id, None)
            self._agents.pop(agent_id, None)
        if remove_output_file:
            try:
                os.remove(snapshot.output_file)
            except FileNotFoundError:
                pass
        return snapshot

    def export_state(self) -> dict[str, Any]:
        snapshots: list[dict[str, Any]] = []
        for snapshot in self.list_snapshots():
            payload = snapshot.to_dict()
            if payload["status"] in {"running", "async_launched", "stop_requested"}:
                finished_at = time.time()
                payload["status"] = "interrupted"
                payload["finishedAt"] = finished_at
                payload["totalDurationMs"] = int((finished_at - snapshot.started_at) * 1000)
                if not payload.get("error"):
                    payload["error"] = "会话恢复后原后台执行上下文不可继续附着，请手动重新启动或续跑。"
            snapshots.append(payload)
        return {
            "version": 1,
            "snapshots": snapshots,
        }

    def restore_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(state or {})
        report: dict[str, Any] = {
            "status": "restored",
            "restoredItems": [],
            "degradedItems": [],
            "missingItems": [],
            "metadata": {},
            "issues": [],
        }
        snapshots: list[SubagentSnapshot] = []
        for item in list(data.get("snapshots") or []):
            if not item:
                continue
            snapshot = SubagentSnapshot.from_dict(item)
            if snapshot.status in {"running", "async_launched", "stop_requested", "waiting"}:
                snapshot.status = "interrupted"
                snapshot.error = snapshot.error or "会话恢复后原后台执行上下文不可继续附着，请手动重新启动或续跑。"
            if snapshot.status == "interrupted":
                report["degradedItems"].append(snapshot.agent_id)
                report["issues"].append(
                    {
                        "code": "background_agent_degraded",
                        "message": f"后台子 agent 无法续跑，已按 interrupted 恢复: {snapshot.agent_id}",
                        "severity": "warning",
                        "metadata": {"agentId": snapshot.agent_id},
                    }
                )
                report["status"] = "degraded"
            else:
                report["restoredItems"].append(snapshot.agent_id)
            snapshots.append(snapshot)
        with self._lock:
            self._snapshots = {snapshot.agent_id: snapshot for snapshot in snapshots if snapshot.agent_id}
            self._futures = {}
            self._agents = {}
        self.last_restore_report = report
        return report

    def close(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "closed",
            "restoredItems": [],
            "degradedItems": [],
            "missingItems": [],
            "metadata": {},
            "issues": [],
        }
        with self._lock:
            active_ids = [
                agent_id for agent_id, snapshot in self._snapshots.items()
                if snapshot.status not in {"completed", "error", "stopped", "cancelled", "interrupted"}
            ]
        if active_ids:
            report["status"] = "degraded"
            report["degradedItems"].extend(active_ids)
            report["issues"].append(
                {
                    "code": "background_agents_still_active",
                    "message": f"关闭 SubagentManager 时仍有后台子 agent 未终止: {active_ids}",
                    "severity": "warning",
                    "metadata": {"agentIds": list(active_ids)},
                }
            )
        self._executor.shutdown(wait=False, cancel_futures=False)
        self.last_close_report = report
        return report


__all__ = ["SubagentRequest", "SubagentSnapshot", "SubagentManager"]
