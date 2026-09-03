"""Claude-style Agent tool for sub-agent orchestration."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeAgentInput
from ..runtime import SubagentManager, SubagentRequest, WorktreeManager
from core.permissions import PermissionContext, PermissionMode
from core.cache_policy import PromptCachePolicy, build_cache_signature
from core.request_compiler import compile_prompt_blocks
from runtime import AgentRuntimeManager, ExecutionContext
from task import TaskStatus
from .bash_tool import BashTool, register_shell_tools
from .display_utils import format_error_display, format_structured_display
from .file_edit import FileEditTool
from .file_write import FileWriteTool
from .filesystem import FileReadTool, GlobTool, GrepTool
from .task_output import TaskOutputTool
from .task_stop import TaskStopTool


AGENT_TOOL_PROMPT = """把一个边界清晰、可独立推进的子任务委派给独立子 agent。

何时使用：
- 当你已经知道要拆出去的子问题，并且它有明确的输入、边界和交付物。
- 当任务可以和当前主线并行推进，或者需要独立的 worktree / mailbox / task 绑定。
- 当你希望把复杂工作分配给多个 agent，再通过 `AgentWait`、`AgentGet` 或 `AgentList` 汇总结果。

如何写好 prompt：
- 明确目标：子 agent 最终必须交付什么，成功标准是什么。
- 明确边界：哪些文件、目录、模块、接口属于它，哪些不要碰。
- 明确约束：是否只读、是否允许改代码、是否允许运行测试、是否必须遵守某种输出格式。
- 明确汇报方式：是直接给出结论，还是修改文件后再总结，还是把结果写入 outputFile 供后续读取。
- 如果你已经知道关键路径、候选文件、函数名或任务上下文，应直接写进 prompt，不要让子 agent盲搜。

后台运行语义：
- `run_in_background=true` 只表示“启动成功并返回 handle”，不表示任务已经完成。
- 启动后应继续用 `AgentGet` / `AgentWait` / `AgentList` 跟踪状态，必要时读取 `outputFile`。
- 如果子 agent 是团队协作成员，后续可以用 `SendMessage` 发运行时消息；子 agent 会通过 mailbox 看到这些消息。

隔离与协作：
- 当子任务需要隔离代码修改或临时 git 分支时，设置 `isolation=worktree`。
- 当多个子 agent 需要共享团队身份时，设置 `team_name`，并先用 `TeamCreate` 建立团队。
- 当父任务已经绑定结构化 task 时，Agent tool 会自动创建 child task 并把 runtime 元数据回写到任务系统。

不要这样用：
- 不要把高度耦合、需要你当前回合立即依赖结果的最关键一步无脑拆出去。
- 不要只给一个笼统标题；prompt 过短会让子 agent 把时间浪费在重新理解需求上。
- 不要把“启动后台 agent”当成“工作已完成”。"""


def _normalize_workspace_root(workspace_root: Optional[str]) -> str:
    return os.path.abspath(workspace_root or os.getcwd())


def _can_read_output_file(registry: Optional[ToolRegistry]) -> bool:
    if registry is None:
        return False
    return registry.has_tool("FileRead") or registry.has_tool("Bash")


def _normalize_subagent_mode(raw_mode: Optional[str]) -> Optional[PermissionMode]:
    mapping = {
        "default": PermissionMode.DEFAULT,
        "plan": PermissionMode.PLAN,
        "dontAsk": PermissionMode.DONT_ASK,
        "acceptEdits": PermissionMode.ACCEPT_EDITS,
        "bypassPermissions": PermissionMode.BYPASS,
    }
    if raw_mode is None:
        return None
    return mapping.get(raw_mode, PermissionMode.DEFAULT)


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
        if name == "skill_tool":
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
        agent_runtime: Optional[AgentRuntimeManager] = None,
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

        factory = self._wrap_agent_factory(
            agent_factory or self._build_default_agent_factory()
        )
        self.agent_factory = factory
        self.agent_runtime = agent_runtime or AgentRuntimeManager(
            agent_factory=factory,
            storage_dir=self.storage_dir,
            max_background_tasks=max_background_tasks,
            subagent_manager=subagent_manager,
        )
        self.subagent_manager = self.agent_runtime.subagent_manager
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
            side_effect_level="medium",
            resource_scope=["runtime", "workspace", "task"],
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

    def _wrap_agent_factory(
        self,
        factory: Callable[[SubagentRequest], Any],
    ) -> Callable[[SubagentRequest], Any]:
        def wrapped(request: SubagentRequest) -> Any:
            child_agent = factory(request)
            parent_observability = getattr(self.parent_agent, "observability", None)
            if (
                parent_observability is not None
                and getattr(child_agent, "observability", None) is None
                and callable(getattr(child_agent, "with_observability", None))
            ):
                child_agent.with_observability(
                    manager=parent_observability.create_child(
                        parent_invoke_id=(request.metadata or {}).get(
                            "parent_agent_invoke_id"
                        ),
                    )
                )
            return child_agent

        return wrapped

    def _delegation_context_metadata(self) -> dict[str, Any]:
        parent_context = getattr(self.parent_agent, "execution_context", None)
        parent_metadata = getattr(parent_context, "metadata", {})
        return {
            "delegatedBy": getattr(self.parent_agent, "name", None),
            "parentAgentId": parent_metadata.get("agentId"),
        }

    def _build_default_agent_factory(self) -> Callable[[SubagentRequest], Any]:
        if self.parent_agent is None:
            raise ValueError("未提供 parent_agent 或 agent_factory，无法构造子 agent。")

        def factory(request: SubagentRequest) -> Any:
            from agent.BasicAgent import BasicAgent

            registry = None
            workspace_root = os.path.abspath(request.workspace_root or self.workspace_root)
            allowed_roots = tuple(request.allowed_roots or (workspace_root,))
            delegated_task_id = None
            if isinstance(request.metadata, dict):
                delegated_task_id = request.metadata.get("task_id")
            parent_registry = getattr(self.parent_agent, "tool_registry", None)
            if self.tool_registry_builder is not None:
                registry = self.tool_registry_builder(request)
            elif parent_registry is not None:
                registry = clone_tool_registry_for_workspace(
                    parent_registry,
                    workspace_root=workspace_root,
                    allowed_roots=allowed_roots,
                    cwd=workspace_root,
                )

            config = getattr(self.parent_agent, "config", None)
            if hasattr(config, "model_copy"):
                config_copy = config.model_copy(deep=True)
            else:
                config_copy = config
            if config_copy is not None:
                config_copy.workspace_root = workspace_root
                config_copy.allowed_roots = list(allowed_roots)
                subagent_cache_policy = str(getattr(config_copy, "subagent_cache_policy", "inherit") or "inherit")
                parent_cache_policy = getattr(config_copy, "cache_policy", None)
                normalized_policy = (
                    PromptCachePolicy.from_value(parent_cache_policy.to_dict())
                    if hasattr(parent_cache_policy, "to_dict")
                    else PromptCachePolicy.from_value(parent_cache_policy)
                )
                if subagent_cache_policy == "read_only":
                    normalized_policy.mode = "read_only"
                    config_copy.cache_policy = normalized_policy
                elif subagent_cache_policy in {"skip_write", "isolated"}:
                    normalized_policy.mode = "skip_write"
                    config_copy.cache_policy = normalized_policy

            parent_permission_context = getattr(self.parent_agent, "permission_context", None)
            if parent_permission_context is not None and hasattr(parent_permission_context, "model_copy"):
                permission_context = parent_permission_context.model_copy(deep=True)
            elif parent_permission_context is not None:
                permission_context = PermissionContext.model_validate(parent_permission_context.model_dump(mode="python"))
            else:
                permission_context = PermissionContext()
            normalized_mode = _normalize_subagent_mode(request.mode)
            if normalized_mode is not None:
                permission_context.set_mode(normalized_mode)

            parent_execution_context = getattr(self.parent_agent, "execution_context", None)
            if parent_execution_context is not None and hasattr(parent_execution_context, "copy_for_workspace"):
                child_execution_context = parent_execution_context.copy_for_workspace(
                    workspace_root=workspace_root,
                    allowed_roots=allowed_roots,
                    worktree_path=request.worktree_path,
                    worktree_branch=request.worktree_branch,
                    execution_mode="plan" if normalized_mode == PermissionMode.PLAN else None,
                    permission_mode=normalized_mode.value if normalized_mode is not None else None,
                    current_task_id=delegated_task_id,
                    metadata=self._delegation_context_metadata(),
                )
            else:
                child_execution_context = ExecutionContext.from_agent(
                    self.parent_agent,
                    workspace_root=workspace_root,
                    allowed_roots=allowed_roots,
                    worktree_path=request.worktree_path,
                    worktree_branch=request.worktree_branch,
                    execution_mode="plan" if normalized_mode == PermissionMode.PLAN else None,
                    permission_mode=normalized_mode.value if normalized_mode is not None else None,
                    current_task_id=delegated_task_id,
                    metadata=self._delegation_context_metadata(),
                )
            child_execution_context.metadata["agentId"] = (request.metadata or {}).get("agent_id")
            child_execution_context.metadata["outputFile"] = (request.metadata or {}).get("output_file")

            child_llm = self.parent_agent.llm
            if request.model and request.model != getattr(child_llm, "model", None):
                from core.llm import EasyLLM

                parent_llm = self.parent_agent.llm
                child_llm = EasyLLM(
                    provider=parent_llm.provider_name,
                    model=request.model,
                    api_key=getattr(parent_llm, "resolve_api_key", None) or parent_llm.api_key,
                    base_url=getattr(parent_llm, "resolve_base_url", None) or parent_llm.base_url,
                    temperature=parent_llm.temperature,
                    max_tokens=parent_llm.max_tokens,
                    timeout=parent_llm.timeout,
                    **dict(getattr(parent_llm, "kwargs", {}) or {}),
                )
            child_agent = BasicAgent(
                name=request.name or request.description or "Subagent",
                llm=child_llm,
                system_prompt=getattr(self.parent_agent, "system_prompt", None),
                description=request.description,
                config=config_copy,
            )
            child_agent.reasoning = getattr(self.parent_agent, "reasoning", None)
            child_agent.execution_context = child_execution_context
            child_agent.with_permissions(
                getattr(self.parent_agent, "permission_engine", None),
                permission_context,
            )
            parent_callbacks = getattr(self.parent_agent, "callback_manager", None)
            if parent_callbacks is not None:
                child_agent.with_callbacks(parent_callbacks)
            if registry is not None:
                child_agent.with_tool(registry)
            parent_skill_manager = getattr(self.parent_agent, "skill_manager", None)
            skill_directories = tuple(
                getattr(parent_skill_manager, "directories", ()) or ()
            )
            if skill_directories:
                child_agent.with_skill(*skill_directories)
            parent_task_service = getattr(self.parent_agent, "task_service", None)
            if parent_task_service is not None:
                child_agent.with_task_service(parent_task_service)
            if normalized_mode == PermissionMode.PLAN:
                from plan import PlanModeConfig

                child_agent.with_plan(config=PlanModeConfig(register_tools=False))
                child_agent.plan.enter()
            if bool(getattr(config_copy, "subagent_inherit_cache_safe_params", True)):
                child_execution_context.metadata.setdefault(
                    "cacheSafeParams",
                    self._build_parent_cache_safe_params(parent_registry),
                )
            if registry is not None and parent_registry is not None:
                collaboration_requested = request.team_name is not None or delegated_task_id is not None or any(
                    parent_registry.has_tool(name)
                    for name in ("Agent", "SendMessage", "MailboxRead", "TeamCreate")
                )
                if collaboration_requested:
                    from runtime import MultiAgentRuntime

                    child_agent.with_multi_agent(
                        MultiAgentRuntime(
                            workspace_root=workspace_root,
                            storage_dir=self.storage_dir,
                            max_background_tasks=self.subagent_manager.max_background_tasks,
                            agent_runtime=self.agent_runtime,
                            team_manager=self.agent_runtime.team_manager,
                        )
                    )
            return child_agent

        return factory

    def _build_parent_cache_safe_params(self, registry: Optional[ToolRegistry]) -> dict[str, Any]:
        parent = self.parent_agent
        try:
            system_blocks = parent.get_system_prompt_blocks()
        except Exception:
            system_blocks = []
        try:
            tools = parent.get_provider_tools() if registry is not None else None
        except Exception:
            tools = None
        cache_policy = getattr(getattr(parent, "config", None), "cache_policy", None)
        try:
            signature = build_cache_signature(
                provider=getattr(getattr(parent, "llm", None), "provider_name", None),
                model=getattr(getattr(parent, "llm", None), "model", None),
                system_blocks=compile_prompt_blocks(
                    system_blocks,
                    cache_policy=cache_policy,
                ).system_prompt_blocks,
                tools=tools,
                reasoning=getattr(parent, "reasoning", None),
                cache_policy=cache_policy,
            ).to_dict()
        except Exception:
            signature = {}
        return {
            "provider": getattr(getattr(parent, "llm", None), "provider_name", None),
            "model": getattr(getattr(parent, "llm", None), "model", None),
            "reasoning": getattr(parent, "reasoning", None),
            "cachePolicy": cache_policy.to_dict() if hasattr(cache_policy, "to_dict") else cache_policy,
            "signature": signature,
        }

    def _build_execution_context(self, request: SubagentRequest) -> ExecutionContext:
        workspace_root = os.path.abspath(request.workspace_root or self.workspace_root)
        allowed_roots = tuple(request.allowed_roots or (workspace_root,))
        normalized_mode = _normalize_subagent_mode(request.mode)
        delegated_task_id = request.metadata.get("task_id") if isinstance(request.metadata, dict) else None
        parent_execution_context = getattr(self.parent_agent, "execution_context", None)
        if parent_execution_context is not None and hasattr(parent_execution_context, "copy_for_workspace"):
            return parent_execution_context.copy_for_workspace(
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                worktree_path=request.worktree_path,
                worktree_branch=request.worktree_branch,
                execution_mode="plan" if normalized_mode == PermissionMode.PLAN else None,
                permission_mode=normalized_mode.value if normalized_mode is not None else None,
                current_task_id=delegated_task_id,
                metadata=self._delegation_context_metadata(),
            )
        if self.parent_agent is not None:
            return ExecutionContext.from_agent(
                self.parent_agent,
                workspace_root=workspace_root,
                allowed_roots=allowed_roots,
                worktree_path=request.worktree_path,
                worktree_branch=request.worktree_branch,
                execution_mode="plan" if normalized_mode == PermissionMode.PLAN else None,
                permission_mode=normalized_mode.value if normalized_mode is not None else None,
                current_task_id=delegated_task_id,
                metadata=self._delegation_context_metadata(),
            )
        return ExecutionContext(
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            execution_mode="plan" if normalized_mode == PermissionMode.PLAN else "execute",
            permission_mode=normalized_mode.value if normalized_mode is not None else "default",
            current_task_id=delegated_task_id,
            worktree_path=request.worktree_path,
            worktree_branch=request.worktree_branch,
            metadata={},
        )

    def _maybe_create_delegation_task(self, request: SubagentRequest) -> Optional[str]:
        task_service = getattr(self.parent_agent, "task_service", None)
        if task_service is None:
            return None
        metadata = dict(request.metadata or {})
        runtime_metadata = dict(metadata.get("runtime") or {})
        runtime_metadata.update(
            {
                "kind": "subagent",
                "delegatedBy": getattr(self.parent_agent, "name", None),
                "teamName": request.team_name,
                "workspaceRoot": request.workspace_root or self.workspace_root,
            }
        )
        metadata["runtime"] = runtime_metadata
        parent_task_id = getattr(self.parent_agent, "current_task_id", None)
        task = task_service.create_task(
            title=request.description,
            description=request.prompt,
            status=TaskStatus.IN_PROGRESS,
            owner=request.name,
            parent_task_id=parent_task_id,
            metadata=metadata,
        )
        request.metadata = {**metadata, "task_id": task.task_id}
        return task.task_id

    def _sync_delegation_task(self, request: SubagentRequest, handle: Any) -> None:
        task_service = getattr(self.parent_agent, "task_service", None)
        task_id = request.metadata.get("task_id") if isinstance(request.metadata, dict) else None
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
        runtime_metadata = dict((request.metadata or {}).get("runtime") or {})
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

    def _prepare_request(self, parameters: dict) -> tuple[SubagentRequest, dict[str, Any]]:
        description = str(parameters.get("description", "")).strip()
        prompt = str(parameters.get("prompt", "")).strip()
        if not description:
            raise ValueError("description 不能为空。")
        if not prompt:
            raise ValueError("prompt 不能为空。")

        metadata: dict[str, Any] = {}
        if bool(getattr(getattr(self.parent_agent, "config", None), "subagent_inherit_cache_safe_params", True)):
            metadata["cacheSafeParams"] = self._build_parent_cache_safe_params(getattr(self.parent_agent, "tool_registry", None))
        metadata["subagentCachePolicy"] = str(
            getattr(getattr(self.parent_agent, "config", None), "subagent_cache_policy", "inherit") or "inherit"
        )
        parent_observability = getattr(self.parent_agent, "observability", None)
        if parent_observability is not None and parent_observability.active_invoke_id:
            metadata["parent_agent_invoke_id"] = parent_observability.active_invoke_id
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
        task_id = self._maybe_create_delegation_task(request)
        if task_id is not None:
            metadata["taskId"] = task_id
        return request, metadata

    def _build_completed_payload(self, handle) -> dict[str, Any]:
        payload = handle.to_tool_payload()
        payload["status"] = handle.status
        return payload

    def _build_async_payload(self, handle) -> dict[str, Any]:
        payload = handle.to_tool_payload()
        payload["status"] = "async_launched"
        payload["canReadOutputFile"] = _can_read_output_file(getattr(self.parent_agent, "tool_registry", None))
        return payload

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
            execution_context = self._build_execution_context(request)
            handle = self.agent_runtime.run(
                request,
                execution_context=execution_context,
                run_in_background=run_in_background,
                agent_factory=self.agent_factory,
            )
            self._sync_delegation_task(request, handle)
            if run_in_background:
                payload = self._build_async_payload(handle)
                return ToolResult.success(
                    f"子 agent 已后台启动: {handle.agent_id}",
                    display_text=format_structured_display(
                        f"子 agent 已后台启动: {handle.agent_id}",
                        payload,
                    ),
                    structured_data=payload,
                    metadata={**payload, **metadata},
                )

            if handle.status == "error":
                error_metadata = {
                    "agentId": handle.agent_id,
                    "outputFile": handle.output_file,
                    **metadata,
                }
                return ToolResult.error(
                    f"子 agent 执行失败: {handle.error}",
                    error_type="subagent_failed",
                    display_text=format_error_display(
                        f"子 agent 执行失败: {handle.error}",
                        error_metadata,
                    ),
                    metadata=error_metadata,
                )

            payload = self._build_completed_payload(handle)
            return ToolResult.success(
                handle.content,
                display_text=format_structured_display(
                    f"子 agent 已完成: {handle.agent_id}",
                    payload,
                    result_text=handle.content,
                ),
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
    agent_runtime: Optional[AgentRuntimeManager] = None,
    subagent_manager: Optional[SubagentManager] = None,
    worktree_manager: Optional[WorktreeManager] = None,
    workspace_root: Optional[str] = None,
    allowed_roots: Optional[Iterable[str]] = None,
    storage_dir: Optional[str] = None,
    max_background_tasks: int = 4,
    expose_in_deferred: bool | None = True,
) -> AgentTool:
    tool = AgentTool(
        parent_agent=parent_agent,
        agent_factory=agent_factory,
        tool_registry_builder=tool_registry_builder,
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


__all__ = ["AgentTool", "clone_tool_registry_for_workspace", "register_agent_tool"]
