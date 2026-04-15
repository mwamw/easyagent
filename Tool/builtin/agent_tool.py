"""Claude-style Agent tool for sub-agent orchestration."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeAgentInput
from ..runtime import SubagentManager, SubagentRequest, WorktreeManager
from .bash_tool import BashTool, register_shell_tools
from .file_edit import FileEditTool
from .file_write import FileWriteTool
from .filesystem import FileReadTool, GlobTool, GrepTool
from .task_output import TaskOutputTool
from .task_stop import TaskStopTool


AGENT_TOOL_PROMPT = """用于把明确的子任务交给独立子 agent。
- prompt 要写完整，明确交付物和边界。
- 当子任务需要隔离代码修改时，设置 `isolation=worktree`。
- 长时间运行的任务可设置 `run_in_background=true`，然后读取 outputFile 查看进度。"""


def _normalize_workspace_root(workspace_root: Optional[str]) -> str:
    return os.path.abspath(workspace_root or os.getcwd())


def _can_read_output_file(registry: Optional[ToolRegistry]) -> bool:
    if registry is None:
        return False
    return registry.has_tool("FileRead") or registry.has_tool("Bash")


def clone_tool_registry_for_workspace(
    registry: ToolRegistry,
    *,
    workspace_root: str,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
) -> ToolRegistry:
    cloned = ToolRegistry()
    shell_tools_registered = False
    visible_names = registry.get_tool_names()

    for name in visible_names:
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
        parent_agent: Any | None = None,
        agent_factory: Optional[Callable[[SubagentRequest], Any]] = None,
        tool_registry_builder: Optional[Callable[[SubagentRequest], Optional[ToolRegistry]]] = None,
        subagent_manager: Optional[SubagentManager] = None,
        worktree_manager: Optional[WorktreeManager] = None,
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        storage_dir: Optional[str] = None,
        max_background_tasks: int = 4,
    ):
        self.parent_agent = parent_agent
        self.workspace_root = _normalize_workspace_root(
            workspace_root
            or getattr(getattr(parent_agent, "config", None), "workspace_root", None)
            or os.getcwd()
        )
        normalized_allowed_roots = tuple(
            os.path.abspath(item)
            for item in (allowed_roots or getattr(getattr(parent_agent, "config", None), "get_allowed_roots", lambda: [])())
            if item
        )
        self.allowed_roots = normalized_allowed_roots or (self.workspace_root,)
        self.tool_registry_builder = tool_registry_builder
        self.worktree_manager = worktree_manager or self._maybe_create_worktree_manager(parent_agent)
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(self.workspace_root, ".easyagent-agents"))

        factory = agent_factory or self._build_default_agent_factory()
        self.subagent_manager = subagent_manager or SubagentManager(
            agent_factory=factory,
            storage_dir=self.storage_dir,
            max_background_tasks=max_background_tasks,
        )
        super().__init__(
            name="Agent",
            description="启动独立子 agent 处理一个明确子任务，支持后台运行和 worktree 隔离。",
            parameters=ClaudeAgentInput,
            guidance="适合把边界清楚、可以独立完成的子任务委派出去。prompt 应包含目标、约束和预期产物。",
            prompt=AGENT_TOOL_PROMPT,
            read_only=False,
            destructive=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "orchestration", "claude_code"],
        )

    def _maybe_create_worktree_manager(self, parent_agent: Any | None) -> Optional[WorktreeManager]:
        config = getattr(parent_agent, "config", None)
        enable_worktree = bool(getattr(config, "enable_worktree", False))
        if not enable_worktree:
            return None
        try:
            repo_root = WorktreeManager.detect_repo_root(self.workspace_root, git_binary=getattr(config, "git_binary", "git"))
        except Exception:
            return None
        return WorktreeManager(
            repo_root,
            git_binary=getattr(config, "git_binary", "git"),
            original_cwd=self.workspace_root,
        )

    def _build_default_agent_factory(self) -> Callable[[SubagentRequest], Any]:
        if self.parent_agent is None:
            raise ValueError("未提供 parent_agent 或 agent_factory，无法构造子 agent。")

        def factory(request: SubagentRequest) -> Any:
            from agent.BasicAgent import BasicAgent

            registry = None
            if self.tool_registry_builder is not None:
                registry = self.tool_registry_builder(request)
            elif getattr(self.parent_agent, "tool_registry", None) is not None:
                parent_registry = self.parent_agent.tool_registry
                if request.workspace_root and os.path.abspath(request.workspace_root) != self.workspace_root:
                    registry = clone_tool_registry_for_workspace(
                        parent_registry,
                        workspace_root=request.workspace_root,
                        allowed_roots=request.allowed_roots or (request.workspace_root,),
                        cwd=request.workspace_root,
                    )
                else:
                    registry = parent_registry

            config = getattr(self.parent_agent, "config", None)
            if hasattr(config, "model_copy"):
                config_copy = config.model_copy(deep=True)
            else:
                config_copy = config
            if config_copy is not None and request.workspace_root:
                config_copy.workspace_root = request.workspace_root
                config_copy.allowed_roots = list(request.allowed_roots or (request.workspace_root,))

            return BasicAgent(
                name=request.name or request.description or "Subagent",
                llm=self.parent_agent.llm,
                system_prompt=getattr(self.parent_agent, "system_prompt", None),
                enable_tool=registry is not None,
                tool_registry=registry,
                description=request.description,
                config=config_copy,
                verbose_thinking=bool(getattr(self.parent_agent, "verbose_thinking", False)),
                callback_manager=getattr(self.parent_agent, "callback_manager", None),
            )

        return factory

    def _prepare_request(self, parameters: dict) -> tuple[SubagentRequest, dict[str, Any]]:
        description = str(parameters.get("description", "")).strip()
        prompt = str(parameters.get("prompt", "")).strip()
        if not description:
            raise ValueError("description 不能为空。")
        if not prompt:
            raise ValueError("prompt 不能为空。")

        metadata: dict[str, Any] = {}
        workspace_root = self.workspace_root
        allowed_roots = self.allowed_roots
        worktree_path = None
        worktree_branch = None

        if parameters.get("isolation") == "worktree":
            if self.worktree_manager is None:
                raise ValueError("当前 Agent tool 未配置 WorktreeManager，无法使用 isolation=worktree。")
            worktree_name = parameters.get("name") or description or "agent-task"
            worktree = self.worktree_manager.create_worktree(str(worktree_name))
            worktree_path = worktree.path
            worktree_branch = worktree.branch
            workspace_root = worktree.path
            allowed_roots = (worktree.path,)
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
        return request, metadata

    def _build_completed_payload(self, snapshot, request: SubagentRequest) -> dict[str, Any]:
        return {
            "status": "completed",
            "agentId": snapshot.agent_id,
            "agentType": request.agent_type,
            "content": [{"type": "text", "text": snapshot.content}],
            "totalToolUseCount": snapshot.total_tool_use_count,
            "totalDurationMs": snapshot.total_duration_ms,
            "totalTokens": snapshot.total_tokens,
            "usage": snapshot.usage,
            "prompt": request.prompt,
            "description": request.description,
            "worktreePath": snapshot.worktree_path,
            "worktreeBranch": snapshot.worktree_branch,
            "outputFile": snapshot.output_file,
        }

    def _build_async_payload(self, snapshot, request: SubagentRequest) -> dict[str, Any]:
        return {
            "status": "async_launched",
            "agentId": snapshot.agent_id,
            "description": request.description,
            "prompt": request.prompt,
            "outputFile": snapshot.output_file,
            "canReadOutputFile": _can_read_output_file(getattr(self.parent_agent, "tool_registry", None)),
            "worktreePath": snapshot.worktree_path,
            "worktreeBranch": snapshot.worktree_branch,
        }

    def run(self, parameters: dict) -> ToolResult:
        try:
            request, metadata = self._prepare_request(parameters)
        except Exception as exc:
            return ToolResult.error(
                f"启动子 agent 失败: {exc}",
                error_type="invalid_parameters",
                metadata={"reason": "invalid_parameters"},
            )

        run_in_background = bool(parameters.get("run_in_background", False))
        try:
            if run_in_background:
                snapshot = self.subagent_manager.launch_background(request)
                payload = self._build_async_payload(snapshot, request)
                return ToolResult.success(
                    f"子 agent 已后台启动: {snapshot.agent_id}",
                    structured_data=payload,
                    metadata={**payload, **metadata},
                )

            snapshot = self.subagent_manager.run(request)
            if snapshot.status == "error":
                return ToolResult.error(
                    f"子 agent 执行失败: {snapshot.error}",
                    error_type="subagent_failed",
                    metadata={
                        "agentId": snapshot.agent_id,
                        "outputFile": snapshot.output_file,
                        **metadata,
                    },
                )

            payload = self._build_completed_payload(snapshot, request)
            return ToolResult.success(
                snapshot.content,
                structured_data=payload,
                metadata={**payload, **metadata},
            )
        except Exception as exc:
            return ToolResult.error(
                f"启动子 agent 失败: {exc}",
                error_type="subagent_failed",
                metadata={"reason": "subagent_failed", **metadata},
            )


def register_agent_tool(
    registry: ToolRegistry,
    *,
    parent_agent: Any | None = None,
    agent_factory: Optional[Callable[[SubagentRequest], Any]] = None,
    tool_registry_builder: Optional[Callable[[SubagentRequest], Optional[ToolRegistry]]] = None,
    subagent_manager: Optional[SubagentManager] = None,
    worktree_manager: Optional[WorktreeManager] = None,
    workspace_root: Optional[str] = None,
    allowed_roots: Optional[Iterable[str]] = None,
    storage_dir: Optional[str] = None,
    max_background_tasks: int = 4,
) -> AgentTool:
    tool = AgentTool(
        parent_agent=parent_agent,
        agent_factory=agent_factory,
        tool_registry_builder=tool_registry_builder,
        subagent_manager=subagent_manager,
        worktree_manager=worktree_manager,
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        storage_dir=storage_dir,
        max_background_tasks=max_background_tasks,
    )
    registry.register_tool(tool)
    return tool


__all__ = ["AgentTool", "clone_tool_registry_for_workspace", "register_agent_tool"]
