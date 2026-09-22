"""Claude Code-style Agent tool backed by an injected child-agent factory."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable

from permissions import ApprovalMode
from runtime import ExecutionContext
from multiagent.default.agents import AgentRuntimeManager
from multiagent.default.subagents import SubagentManager, SubagentRequest
from task import TaskStatus

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.claude_compat.models import ClaudeAgentInput
from Tool.runtime import WorktreeManager
from Tool.builtin.bash_tool import BashTool, register_shell_tools
from Tool.builtin.display_utils import format_error_display, format_structured_display
from Tool.builtin.file_edit import FileEditTool
from Tool.builtin.file_write import FileWriteTool
from Tool.builtin.filesystem import FileReadTool, GlobTool, GrepTool
from Tool.builtin.task_output import TaskOutputTool
from Tool.builtin.task_stop import TaskStopTool


AGENT_TOOL_PROMPT = """Launch a new agent to handle complex, multi-step tasks autonomously.

The Agent tool launches specialized agents that autonomously handle complex tasks. Each fresh invocation starts without conversation context, so provide a complete task description.

When NOT to use Agent:
- If you want to read a specific file path, use FileRead or Glob instead.
- If you are searching for a specific class or symbol, use Grep.
- If you are searching within a known set of 2-3 files, use FileRead or Grep.
- Do not use an agent for work unrelated to the available agent descriptions.

Usage notes:
- Always include a short description (3-5 words) summarizing what the agent will do.
- Launch multiple agents concurrently whenever possible by emitting independent Agent tool calls in one response.
- The result is returned to you, not directly to the user. Summarize it for the user.
- Use foreground when you need the result before proceeding. Use run_in_background only for genuinely independent work; completion is delivered asynchronously, so do not sleep or poll.
- Clearly tell the agent whether it should modify files or only research.
- Avoid duplicating work already delegated to a subagent.
- Set isolation="worktree" when code changes need an isolated repository checkout.

## Writing the prompt

Brief the agent like a smart colleague who just walked into the room: it has not seen this conversation, does not know what you tried, and does not know why the task matters.
- Explain what you are trying to accomplish and why.
- Describe what you already learned or ruled out.
- Give enough surrounding context for judgment rather than prescribing brittle steps.
- If you need a short response, say so.
- For lookups, provide the exact command. For investigations, provide the question.

Terse command-style prompts produce shallow, generic work. Never delegate understanding with phrases such as "based on your findings, fix the bug". Include the concrete paths, constraints, evidence, and requested change."""


MULTI_AGENT_TOOL_NAMES = frozenset(
    {
        "Agent",
        "AgentGet",
        "AgentList",
        "AgentWait",
        "AgentStop",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
    }
)


def _normalize_workspace_root(workspace_root: str | None) -> str:
    return os.path.abspath(workspace_root or os.getcwd())


def _can_read_output_file(registry: ToolRegistry | None) -> bool:
    return registry is not None and (registry.has_tool("FileRead") or registry.has_tool("Bash"))


def _normalize_subagent_mode(raw_mode: str | None) -> ApprovalMode | None:
    mapping = {
        "default": ApprovalMode.ASK_FOR_APPROVAL,
        "dontAsk": ApprovalMode.FULL_ACCESS,
        "acceptEdits": ApprovalMode.AUTO_APPROVE_SAFE,
        "bypassPermissions": ApprovalMode.FULL_ACCESS,
    }
    if raw_mode is None:
        return None
    return mapping.get(raw_mode, ApprovalMode.ASK_FOR_APPROVAL)


def clone_tool_registry_for_workspace(
    registry: ToolRegistry,
    *,
    workspace_root: str,
    allowed_roots: Iterable[str] | None = None,
    cwd: str | None = None,
) -> ToolRegistry:
    """Clone workspace-aware tools without copying collaboration bindings."""

    cloned = ToolRegistry()
    shell_tools_registered = False
    for name in registry.get_tool_names():
        if name in MULTI_AGENT_TOOL_NAMES or name == "skill_tool":
            continue
        tool = registry.get_tool(name)
        if tool is None:
            continue
        visibility = registry.get_tool_visibility(name) or "resident"
        if isinstance(tool, BashTool):
            if shell_tools_registered:
                continue
            register_shell_tools(
                cloned,
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                cwd=cwd or workspace_root,
                shell=tool.process_manager.shell,
                command_timeout_ms=tool.command_timeout_ms,
                max_background_tasks=tool.process_manager.max_background_tasks,
                max_output_chars=tool.max_output_chars,
                sandbox=tool.sandbox_config,
            )
            shell_tools_registered = True
            continue
        if isinstance(tool, (TaskOutputTool, TaskStopTool)):
            continue
        if isinstance(tool, FileReadTool):
            cloned_tool = FileReadTool(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                cwd=cwd or workspace_root,
                max_output_chars=tool.max_output_chars,
            )
        elif isinstance(tool, GlobTool):
            cloned_tool = GlobTool(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                cwd=cwd or workspace_root,
                max_results=tool.max_results,
            )
        elif isinstance(tool, GrepTool):
            cloned_tool = GrepTool(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                cwd=cwd or workspace_root,
                rg_binary=tool.rg_binary,
                max_head_limit=tool.max_head_limit,
                max_output_chars=tool.max_output_chars,
            )
        elif isinstance(tool, FileWriteTool):
            cloned_tool = FileWriteTool(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                cwd=cwd or workspace_root,
            )
        elif isinstance(tool, FileEditTool):
            cloned_tool = FileEditTool(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                cwd=cwd or workspace_root,
            )
        else:
            cloned_tool = tool
        cloned.register_tool(cloned_tool, visibility=visibility)
    return cloned


class AgentTool(Tool):
    def __init__(
        self,
        *,
        agent_factory: Callable[[SubagentRequest], Any],
        execution_context: ExecutionContext | None = None,
        tool_registry: ToolRegistry | None = None,
        agent_name: str | None = None,
        task_service: Any = None,
        observability: Any = None,
        cache_safe_params_provider: Callable[[], dict[str, Any]] | None = None,
        agent_runtime: AgentRuntimeManager | None = None,
        subagent_manager: SubagentManager | None = None,
        worktree_manager: WorktreeManager | None = None,
        workspace_root: str | None = None,
        allowed_roots: Iterable[str] | None = None,
        storage_dir: str | None = None,
        max_background_tasks: int = 4,
    ) -> None:
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self.agent_factory = agent_factory
        self.execution_context = execution_context
        self.tool_registry = tool_registry
        self.agent_name = agent_name
        self.task_service = task_service
        self.observability = observability
        self.cache_safe_params_provider = cache_safe_params_provider
        self.workspace_root = _normalize_workspace_root(
            workspace_root or getattr(execution_context, "workspace_root", None)
        )
        roots = tuple(os.path.abspath(item) for item in (allowed_roots or ()) if item)
        self.allowed_roots = roots or tuple(getattr(execution_context, "allowed_roots", ())) or (self.workspace_root,)
        self.worktree_manager = worktree_manager
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(self.workspace_root, ".easyagent-agents"))
        self.agent_runtime = agent_runtime or AgentRuntimeManager(
            agent_factory=agent_factory,
            storage_dir=self.storage_dir,
            max_background_tasks=max_background_tasks,
            subagent_manager=subagent_manager,
        )
        self.subagent_manager = self.agent_runtime.subagent_manager
        super().__init__(
            name="Agent",
            description="Launch a specialized agent for a scoped task.",
            parameters=ClaudeAgentInput,
            guidance="Provide a complete goal, context, constraints, and expected deliverable.",
            prompt=AGENT_TOOL_PROMPT,
            read_only=False,
            destructive=False,
            supports_parallel=True,
            source="builtin",
            tags=["agent", "orchestration", "claude_code"],
            side_effect_level="medium",
            resource_scope=["runtime", "workspace", "task"],
        )

    def _cache_safe_params(self) -> dict[str, Any]:
        if self.cache_safe_params_provider is None:
            return {}
        try:
            return dict(self.cache_safe_params_provider())
        except Exception:
            return {}

    def _build_execution_context(self, request: SubagentRequest) -> ExecutionContext:
        workspace_root = os.path.abspath(request.workspace_root or self.workspace_root)
        allowed_roots = tuple(request.allowed_roots or (workspace_root,))
        mode = _normalize_subagent_mode(request.mode)
        task_id = request.metadata.get("task_id") if isinstance(request.metadata, dict) else None
        metadata = {
            "delegatedBy": self.agent_name,
            "parentAgentId": (
                self.execution_context.metadata.get("agentId")
                if self.execution_context is not None
                else None
            ),
        }
        if self.execution_context is not None:
            return self.execution_context.copy_for_workspace(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                worktree_path=request.worktree_path,
                worktree_branch=request.worktree_branch,
                permission_mode=mode.value if mode is not None else None,
                current_task_id=task_id,
                metadata=metadata,
            )
        return ExecutionContext(
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            permission_mode=mode.value if mode is not None else "default",
            current_task_id=task_id,
            worktree_path=request.worktree_path,
            worktree_branch=request.worktree_branch,
            metadata=metadata,
        )

    def _create_task(self, request: SubagentRequest) -> str | None:
        if self.task_service is None:
            return None
        metadata = dict(request.metadata or {})
        metadata["runtime"] = {
            **dict(metadata.get("runtime") or {}),
            "kind": "subagent",
            "delegatedBy": self.agent_name,
            "teamName": request.team_name,
            "workspaceRoot": request.workspace_root or self.workspace_root,
        }
        task = self.task_service.create_task(
            title=request.description,
            description=request.prompt,
            status=TaskStatus.IN_PROGRESS,
            owner=request.name,
            parent_task_id=getattr(self.execution_context, "current_task_id", None),
            metadata=metadata,
        )
        request.metadata = {**metadata, "task_id": task.task_id}
        return task.task_id

    def _sync_task(self, request: SubagentRequest, handle: Any) -> None:
        task_id = request.metadata.get("task_id") if isinstance(request.metadata, dict) else None
        if self.task_service is None or not task_id:
            return
        statuses = {
            "completed": TaskStatus.COMPLETED,
            "error": TaskStatus.BLOCKED,
            "stopped": TaskStatus.CANCELLED,
            "interrupted": TaskStatus.BLOCKED,
            "stop_requested": TaskStatus.BLOCKED,
            "async_launched": TaskStatus.IN_PROGRESS,
            "running": TaskStatus.IN_PROGRESS,
        }
        runtime = {
            **dict((request.metadata or {}).get("runtime") or {}),
            "agentId": handle.agent_id,
            "teamId": getattr(handle, "team_id", None),
            "teamName": getattr(handle, "team_name", None),
            "outputFile": handle.output_file,
            "status": handle.status,
        }
        try:
            self.task_service.update_task(
                task_id,
                owner=handle.agent_id,
                status=statuses.get(handle.status, TaskStatus.IN_PROGRESS),
                metadata={"runtime": runtime},
            )
        except Exception:
            return

    def _prepare_request(self, parameters: dict[str, Any]) -> tuple[SubagentRequest, dict[str, Any]]:
        description = str(parameters.get("description") or "").strip()
        prompt = str(parameters.get("prompt") or "").strip()
        if not description:
            raise ValueError("description must not be empty")
        if not prompt:
            raise ValueError("prompt must not be empty")
        metadata: dict[str, Any] = {"cacheSafeParams": self._cache_safe_params()}
        if self.observability is not None and self.observability.active_invoke_id:
            metadata["parent_agent_invoke_id"] = self.observability.active_invoke_id
        workspace_root = self.workspace_root
        allowed_roots = self.allowed_roots
        worktree_path = None
        worktree_branch = None
        if parameters.get("isolation") == "worktree":
            if self.worktree_manager is None:
                raise ValueError("isolation=worktree requires a WorktreeManager")
            worktree = self.worktree_manager.create_worktree(
                str(parameters.get("name") or description or "agent-task")
            )
            workspace_root = worktree.path
            allowed_roots = (worktree.path,)
            worktree_path = worktree.path
            worktree_branch = worktree.branch
            metadata["worktree_created"] = True
        request = SubagentRequest(
            description=description,
            prompt=prompt,
            agent_type=parameters.get("subagent_type"),
            model=parameters.get("model"),
            name=parameters.get("name"),
            team_name=parameters.get("team_name"),
            mode=parameters.get("mode"),
            isolation=parameters.get("isolation"),
            workspace_root=workspace_root,
            allowed_roots=tuple(allowed_roots),
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            metadata=metadata,
        )
        task_id = self._create_task(request)
        if task_id is not None:
            metadata["taskId"] = task_id
        return request, metadata

    def run(self, parameters: dict) -> ToolResult:
        try:
            request, metadata = self._prepare_request(parameters)
            handle = self.agent_runtime.run(
                request,
                execution_context=self._build_execution_context(request),
                run_in_background=bool(parameters.get("run_in_background", False)),
                agent_factory=self.agent_factory,
            )
            self._sync_task(request, handle)
        except Exception as exc:
            return ToolResult.error(
                f"Failed to launch subagent: {exc}",
                error_type="subagent_failed",
                metadata={"reason": "subagent_failed"},
            )
        payload = handle.to_tool_payload()
        if bool(parameters.get("run_in_background", False)):
            payload["status"] = "async_launched"
            payload["canReadOutputFile"] = _can_read_output_file(self.tool_registry)
            message = f"Subagent started in background: {handle.agent_id}"
        elif handle.status == "error":
            error_metadata = {"agentId": handle.agent_id, "outputFile": handle.output_file, **metadata}
            return ToolResult.error(
                f"Subagent failed: {handle.error}",
                error_type="subagent_failed",
                display_text=format_error_display(f"Subagent failed: {handle.error}", error_metadata),
                metadata=error_metadata,
            )
        else:
            payload["status"] = handle.status
            message = f"Subagent completed: {handle.agent_id}"
        return ToolResult.success(
            handle.content if handle.status == "completed" else message,
            display_text=format_structured_display(message, payload, result_text=handle.content),
            structured_data=payload,
            metadata={**payload, **metadata},
        )


def register_agent_tool(
    registry: ToolRegistry,
    *,
    agent_factory: Callable[[SubagentRequest], Any],
    execution_context: ExecutionContext | None = None,
    tool_registry: ToolRegistry | None = None,
    agent_name: str | None = None,
    task_service: Any = None,
    observability: Any = None,
    cache_safe_params_provider: Callable[[], dict[str, Any]] | None = None,
    agent_runtime: AgentRuntimeManager | None = None,
    subagent_manager: SubagentManager | None = None,
    worktree_manager: WorktreeManager | None = None,
    workspace_root: str | None = None,
    allowed_roots: Iterable[str] | None = None,
    storage_dir: str | None = None,
    max_background_tasks: int = 4,
    expose_in_deferred: bool | None = True,
) -> AgentTool:
    tool = AgentTool(
        agent_factory=agent_factory,
        execution_context=execution_context,
        tool_registry=tool_registry or registry,
        agent_name=agent_name,
        task_service=task_service,
        observability=observability,
        cache_safe_params_provider=cache_safe_params_provider,
        agent_runtime=agent_runtime,
        subagent_manager=subagent_manager,
        worktree_manager=worktree_manager,
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        storage_dir=storage_dir,
        max_background_tasks=max_background_tasks,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = [
    "AGENT_TOOL_PROMPT",
    "AgentTool",
    "MULTI_AGENT_TOOL_NAMES",
    "clone_tool_registry_for_workspace",
    "register_agent_tool",
]
