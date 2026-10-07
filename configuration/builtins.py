"""Configuration recipes for shipped implementations, not new module contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .errors import ConfigurationError
from .models import ConfigurationPaths, Options
from .registry import BuildContext, ConfigurationRegistry


class ModelOptions(Options):
    provider: str = "openai"
    model: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: int | None = Field(default=None, gt=0)
    default_headers: dict[str, str] = Field(default_factory=dict, repr=False)
    provider_options: dict[str, Any] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def disjoint_provider_options(self):
        if set(self.provider_options) & set(type(self).model_fields):
            raise ValueError("provider_options cannot override explicit model fields")
        return self


class ContextOptions(Options):
    max_tokens: int | None = Field(default=None, gt=0)


class PromptOptions(Options):
    include_defaults: bool = True
    include_module_blocks: bool = True


class PermissionOptions(Options):
    approval_mode: Literal["ask_for_approval", "auto_approve_safe", "full_access"] = "ask_for_approval"
    policy_directory: str | None = None
    policy_paths: list[str] = Field(default_factory=list)


class MemoryOptions(Options):
    user_id: str = "default_user"
    max_capacity: int = Field(default=15, gt=0)
    max_working_token: int = Field(default=10000, gt=0)
    importance_threshold: float = Field(default=0.1, ge=0, le=1)
    decay_factor: float = Field(default=0.95, ge=0, le=1)
    batch_size: int = Field(default=50, gt=0)


class PlanOptions(Options):
    allowed_actions: list[str] = Field(default_factory=list)
    register_tools: bool = True
    enter_message: str | None = None
    exit_message: str | None = None


class PathOptions(Options):
    path: str


class MultiAgentOptions(Options):
    storage_dir: str = ".easyagent/agents"
    max_background_tasks: int = Field(default=4, gt=0)


class WorktreeOptions(Options):
    storage_dir: str = ".easyagent/worktrees"
    git_binary: str = "git"


class WorkspaceToolOptions(Options):
    workspace_root: str | None = None
    allowed_roots: list[str] | None = None
    cwd: str | None = None
    expose_in_deferred: bool = True


class FilesystemOptions(WorkspaceToolOptions):
    rg_binary: str | None = None


class SandboxOptions(Options):
    enabled: bool = False
    mode: Literal["regular", "auto_allow"] = "regular"
    allow_unsandboxed_commands: bool = True
    fail_if_unavailable: bool = False
    network: Literal["inherit", "deny"] = "deny"
    allow_write: list[str] = Field(default_factory=list)
    deny_write: list[str] = Field(default_factory=list)
    allow_read: list[str] = Field(default_factory=list)
    deny_read: list[str] = Field(default_factory=list)
    excluded_commands: list[str] = Field(default_factory=list)
    protect_git: bool = True
    bubblewrap_path: str | None = None


class BashOptions(WorkspaceToolOptions):
    shell: str = "bash"
    command_timeout_ms: int = Field(default=120000, gt=0)
    max_background_tasks: int = Field(default=8, gt=0)
    max_output_chars: int = Field(default=30000, gt=0)
    sandbox: SandboxOptions = Field(default_factory=SandboxOptions)


class SearchOptions(Options):
    api_key: str | None = Field(default=None, repr=False)
    backend: Literal["auto", "serpapi", "duckduckgo"] = "auto"


class FetchOptions(Options):
    request_timeout_s: float = Field(default=20, gt=0)
    max_content_chars: int = Field(default=40000, gt=0)
    max_display_chars: int = Field(default=12000, gt=0)


def _path_options(options, paths: ConfigurationPaths):
    data = options.model_dump()
    for name in ("path", "storage_dir", "policy_directory"):
        if data.get(name) is not None:
            data[name] = paths.resolve(data[name])
    if "policy_paths" in data:
        data["policy_paths"] = [paths.resolve(p) for p in data["policy_paths"]]
        for value in data["policy_paths"]:
            if not Path(value).is_file():
                raise ConfigurationError("permissions.config.policy_paths", "Policy file does not exist")
    if data.get("policy_directory") and not Path(data["policy_directory"]).is_dir():
        raise ConfigurationError("permissions.config.policy_directory", "Policy directory does not exist")
    if "policy_paths" in data:
        from permissions.default.policy import load_policy_file, policy_locations
        for path in policy_locations(policy_directory=data.get("policy_directory"), policy_paths=data["policy_paths"]):
            try:
                load_policy_file(path)
            except Exception:
                raise ConfigurationError("permissions.config", "Invalid or unreadable policy JSON") from None
    return type(options)(**data)


def _workspace_options(options, paths: ConfigurationPaths):
    data = options.model_dump()
    data["workspace_root"] = paths.resolve(data["workspace_root"]) if data["workspace_root"] else str(paths.workspace)
    data["cwd"] = paths.resolve(data["cwd"]) if data["cwd"] else data["workspace_root"]
    data["allowed_roots"] = (
        [paths.resolve(p) for p in data["allowed_roots"]]
        if data["allowed_roots"] is not None else list(paths.allowed_roots)
    )
    for name in ("workspace_root", "cwd"):
        if not Path(data[name]).is_dir():
            raise ConfigurationError(f"tools.config.{name}", "Directory does not exist")
    if "sandbox" in data:
        for name in ("allow_write", "deny_write", "allow_read", "deny_read"):
            data["sandbox"][name] = [paths.resolve(p) for p in data["sandbox"][name]]
        if data["sandbox"]["bubblewrap_path"]:
            data["sandbox"]["bubblewrap_path"] = paths.resolve(data["sandbox"]["bubblewrap_path"])
    return type(options)(**data)


def default_registry() -> ConfigurationRegistry:
    """Return a fresh registry; importing this module does not load Memory/MCP."""
    from core.llm import EasyLLM
    from context import ContextManager
    from permissions import RuleBasedPermissionEngine
    from prompt.composer import SystemPromptComposer
    from executor import DefaultAgentExecutor

    registry = ConfigurationRegistry()

    def model(options: ModelOptions, paths: ConfigurationPaths) -> EasyLLM:
        values = options.model_dump(exclude={"provider_options"})
        return EasyLLM(**values, **options.provider_options)

    registry.register_model("default", config_model=ModelOptions, factory=model)
    registry.register_context("default", config_model=ContextOptions,
                              factory=lambda o, c: ContextManager(**o.model_dump()))
    registry.register_executor("default", config_model=Options,
                               factory=lambda o, c: DefaultAgentExecutor())
    def prompt(agent, options: PromptOptions, context):
        from prompt import PromptBlock
        blocks = [PromptBlock("identity", agent.system_prompt)] if (
            not options.include_defaults and agent.system_prompt is not None
        ) else []
        agent.with_prompt(SystemPromptComposer(blocks=blocks, **options.model_dump()))

    registry.register_installer("prompt", "default", config_model=PromptOptions, install=prompt)
    for name in ("default", "rule_based"):
        registry.register_permission(
            name, config_model=PermissionOptions, prepare=_path_options,
            factory=lambda o, c: RuleBasedPermissionEngine(
                workspace_root=str(c.paths.workspace), **o.model_dump(),
            ),
        )

    def memory(options: MemoryOptions, context: BuildContext):
        from memory import MemoryManage, MemoryConfig
        return MemoryManage(
            user_id=options.user_id,
            config=MemoryConfig(**options.model_dump(exclude={"user_id"})),
        )

    registry.register_memory("default", config_model=MemoryOptions, factory=memory)

    def plan(agent, options: PlanOptions, context):
        from plan import PlanModeConfig
        agent.with_plan(config=PlanModeConfig(**options.model_dump(exclude_none=True)))

    def tasks(agent, options, context):
        from task import TaskService, InMemoryTaskStore
        agent.with_task_service(TaskService(InMemoryTaskStore()))

    def session(agent, options, context):
        from session import SQLiteSessionStore, InMemorySessionStore
        store = SQLiteSessionStore(options.path) if isinstance(options, PathOptions) else InMemorySessionStore()
        try:
            agent.with_session(store, owned=True)
        except BaseException:
            store.close()
            raise

    def observability(agent, options, context):
        from observability import ObservabilityManager, InMemoryObservabilityStore
        if isinstance(options, PathOptions):
            agent.with_observability(path=options.path)
        else:
            agent.with_observability(ObservabilityManager(InMemoryObservabilityStore()))

    def worktree(agent, options: WorktreeOptions, context):
        from worktree import WorktreeManager
        root = WorktreeManager.detect_repo_root(str(context.paths.workspace), git_binary=options.git_binary)
        manager = WorktreeManager(root, storage_dir=options.storage_dir,
                                  original_cwd=str(context.paths.workspace), git_binary=options.git_binary)
        try:
            agent.with_worktree(manager)
        except BaseException:
            manager.close()
            raise

    registry.register_installer("plan", "default", config_model=PlanOptions, install=plan)
    registry.register_installer("tasks", "default", config_model=Options, install=tasks)
    for kind, install in (("session", session), ("observability", observability)):
        registry.register_installer(kind, "default", config_model=Options, install=install)
        registry.register_installer(kind, "sqlite", config_model=PathOptions, install=install, prepare=_path_options)
    registry.register_installer("worktree", "default", config_model=WorktreeOptions,
                                install=worktree, prepare=_path_options)
    registry.register_installer("multi_agent", "default", config_model=MultiAgentOptions,
                                install=lambda a, o, c: a.with_multi_agent(**o.model_dump()), prepare=_path_options)
    registry.register_installer("hooks", "default", config_model=Options, install=lambda a, o, c: a.with_hooks())
    registry.register_installer("callbacks", "default", config_model=Options, install=lambda a, o, c: a.with_callbacks())
    registry.register_installer("codeintel", "default", config_model=Options, install=lambda a, o, c: a.with_codeintel())

    def filesystem(target, options, context):
        from Tool.builtin import register_filesystem_tools
        register_filesystem_tools(target, **options.model_dump())

    def bash(target, options: BashOptions, context):
        from Tool.builtin import register_shell_tools, BashSandboxConfig
        register_shell_tools(target, **options.model_dump(exclude={"sandbox"}),
                             sandbox=BashSandboxConfig(**options.sandbox.model_dump()))

    registry.register_tools("filesystem", config_model=FilesystemOptions, register=filesystem,
                            prepare=_workspace_options, tool_names=("FileRead", "List", "Glob", "Grep"))
    registry.register_tools("bash", config_model=BashOptions, register=bash,
                            prepare=_workspace_options, tool_names=("Bash", "TaskOutput", "TaskStop"))

    def write(target, options, context):
        from Tool.builtin import register_file_write_tool
        register_file_write_tool(target, **options.model_dump())

    def edit(target, options, context):
        from Tool.builtin import register_file_edit_tool
        register_file_edit_tool(target, **options.model_dump())

    def notebook(target, options, context):
        from Tool.builtin import register_notebook_edit_tool
        register_notebook_edit_tool(target, **options.model_dump())

    for name, register in (("file_write", write), ("file_edit", edit), ("notebook_edit", notebook)):
        registry.register_tools(name, config_model=WorkspaceToolOptions, register=register, prepare=_workspace_options)

    def search(target, options, context):
        from Tool.builtin.search import WebSearchTool
        target.register_tool(WebSearchTool(**options.model_dump()))

    def fetch(target, options, context):
        from Tool.builtin.web_fetch import WebFetchTool
        target.register_tool(WebFetchTool(**options.model_dump()))

    def calculator(target, options, context):
        from Tool.builtin.calculator import CalculatorTool
        target.register_tool(CalculatorTool())

    def ask(target, options, context):
        from Tool.builtin.interaction_tools import AskUserQuestionTool
        target.register_tool(AskUserQuestionTool())

    for name, options, register in (
        ("web_search", SearchOptions, search), ("web_fetch", FetchOptions, fetch),
        ("calculator", Options, calculator), ("ask_user_question", Options, ask),
    ):
        registry.register_tools(name, config_model=options, register=register)
    return registry
