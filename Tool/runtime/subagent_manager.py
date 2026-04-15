"""Runtime manager for Claude-style sub-agent execution."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4


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
            "totalDurationMs": self.total_duration_ms,
            "totalToolUseCount": self.total_tool_use_count,
            "totalTokens": self.total_tokens,
            "usage": dict(self.usage),
        }


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
        os.makedirs(self.storage_dir, exist_ok=True)

    def _build_output_file(self, agent_id: str) -> str:
        return os.path.join(self.storage_dir, f"{agent_id}.md")

    def _write_output_file(self, path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

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
        return sum(1 for event in trace_history if isinstance(event, dict) and event.get("type") == "tool_call")

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

        if isinstance(context_usage, dict):
            input_tokens = int(context_usage.get("used_tokens") or 0)
            usage["input_tokens"] = input_tokens
        else:
            input_tokens = 0

        try:
            trace_history = list(getattr(agent, "get_trace_history")())
        except Exception:
            trace_history = list(getattr(agent, "trace_history", []) or [])

        for event in trace_history:
            if not isinstance(event, dict) or event.get("type") != "tool_call":
                continue
            tool_name = str(event.get("tool_name") or "")
            if tool_name == "WebSearch":
                usage["server_tool_use"]["web_search_requests"] += 1
            elif tool_name == "WebFetch":
                usage["server_tool_use"]["web_fetch_requests"] += 1

        total_tokens = int(usage["input_tokens"] or 0) + int(usage["output_tokens"] or 0)
        return usage, total_tokens

    def _execute(self, snapshot: SubagentSnapshot, request: SubagentRequest) -> SubagentSnapshot:
        try:
            agent = self.agent_factory(request)
            result = agent.invoke(request.prompt)
            usage, total_tokens = self._build_usage(agent)
            total_tool_use_count = self._count_tool_calls(agent)
            content = str(result)
            finished_at = time.time()
            output_lines = [
                f"# {request.description or snapshot.agent_id}",
                "",
                f"状态: completed",
                f"开始时间: {snapshot.started_at}",
                f"结束时间: {finished_at}",
                "",
                "## Prompt",
                request.prompt,
                "",
                "## Result",
                content,
            ]
            self._write_output_file(snapshot.output_file, "\n".join(output_lines).strip() + "\n")
            with self._lock:
                snapshot.status = "completed"
                snapshot.content = content
                snapshot.finished_at = finished_at
                snapshot.total_duration_ms = int((finished_at - snapshot.started_at) * 1000)
                snapshot.total_tool_use_count = total_tool_use_count
                snapshot.total_tokens = total_tokens
                snapshot.usage = usage
            return snapshot
        except Exception as exc:
            finished_at = time.time()
            error_text = str(exc)
            self._write_output_file(
                snapshot.output_file,
                (
                    f"# {request.description or snapshot.agent_id}\n\n"
                    f"状态: error\n"
                    f"开始时间: {snapshot.started_at}\n"
                    f"结束时间: {finished_at}\n\n"
                    f"## Prompt\n{request.prompt}\n\n"
                    f"## Error\n{error_text}\n"
                ),
            )
            with self._lock:
                snapshot.status = "error"
                snapshot.error = error_text
                snapshot.finished_at = finished_at
                snapshot.total_duration_ms = int((finished_at - snapshot.started_at) * 1000)
            return snapshot

    def run(self, request: SubagentRequest) -> SubagentSnapshot:
        snapshot = self._create_snapshot(request, status="running")
        self._write_output_file(
            snapshot.output_file,
            f"# {request.description or snapshot.agent_id}\n\n状态: running\n\n## Prompt\n{request.prompt}\n",
        )
        return self._execute(snapshot, request)

    def launch_background(self, request: SubagentRequest) -> SubagentSnapshot:
        snapshot = self._create_snapshot(request, status="async_launched")
        self._write_output_file(
            snapshot.output_file,
            (
                f"# {request.description or snapshot.agent_id}\n\n"
                "状态: async_launched\n\n"
                "## Prompt\n"
                f"{request.prompt}\n"
            ),
        )

        future = self._executor.submit(self._execute, snapshot, request)
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
            future.result()
        return snapshot

    def list_snapshots(self) -> list[SubagentSnapshot]:
        with self._lock:
            agent_ids = list(self._snapshots.keys())
        return [self.get_snapshot(agent_id) for agent_id in agent_ids]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


__all__ = ["SubagentRequest", "SubagentSnapshot", "SubagentManager"]
