"""Explicit child-agent factory used by the built-in multi-agent harness."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from core.cache_policy import PromptCachePolicy, build_cache_signature
from permissions import ApprovalMode
from permissions.default import RuleBasedPermissionEngine
from core.request_compiler import compile_prompt_blocks
from runtime import AgentRuntime
from permissions.base import BasePermissionEngine
from observability.base import BaseObservabilityManager
from plan.base import BasePlanMode
from multiagent.default.definitions import AgentDefinitionRegistry
from .permissions import ChildPermissionFactory, inherit_permission_engine
from .tool_binding import ChildToolFactory, apply_child_tool_factories, clone_tool_registry_for_workspace

if TYPE_CHECKING:
    from Tool.BaseTool import Tool
    from Tool.ToolRegistry import ToolRegistry
    from multiagent.default.agents import AgentRuntimeManager
    from multiagent.default.subagents import SubagentRequest
    from multiagent.default.teams import TeamManager


def _subagent_approval_mode(raw_mode: str | None) -> ApprovalMode | None:
    mapping = {
        "default": ApprovalMode.ASK_FOR_APPROVAL,
        "dontAsk": ApprovalMode.FULL_ACCESS,
        "acceptEdits": ApprovalMode.AUTO_APPROVE_SAFE,
        "bypassPermissions": ApprovalMode.FULL_ACCESS,
    }
    if raw_mode is None:
        return None
    return mapping.get(raw_mode, ApprovalMode.ASK_FOR_APPROVAL)


class DefaultSubagentFactory:
    """Create child Agents from explicitly injected parent capabilities.

    This object intentionally does not retain the parent Agent. It captures the
    modules and immutable product settings required to assemble a child.
    """

    def __init__(
        self,
        *,
        parent_name: str,
        llm: Any,
        config: Any,
        system_prompt: str | None,
        reasoning: dict[str, Any] | None,
        runtime: AgentRuntime,
        tool_registry: "ToolRegistry",
        definitions: AgentDefinitionRegistry,
        permission_engine: BasePermissionEngine | None = None,
        child_permission_factory: ChildPermissionFactory | None = None,
        child_tool_factories: Mapping[str, ChildToolFactory] | None = None,
        parent_bound_tools: Sequence[Tool] = (),
        callback_manager: Any = None,
        skill_directories: tuple[str, ...] = (),
        task_service: Any = None,
        observability: BaseObservabilityManager | None = None,
        plan: BasePlanMode | None = None,
        prompt_blocks_provider: Callable[[], list[Any]] | None = None,
        provider_tools_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.parent_name = parent_name
        self.llm = llm
        self.config = config
        self.system_prompt = system_prompt
        self.reasoning = reasoning
        self.runtime = runtime
        self.tool_registry = tool_registry
        self.permission_engine = permission_engine
        self.child_permission_factory = child_permission_factory or inherit_permission_engine
        self.child_tool_factories = dict(child_tool_factories or {})
        self.parent_bound_tools = tuple(parent_bound_tools)
        self.callback_manager = callback_manager
        self.skill_directories = tuple(skill_directories)
        self.task_service = task_service
        self.observability = observability
        self.plan = plan
        self.prompt_blocks_provider = prompt_blocks_provider
        self.provider_tools_provider = provider_tools_provider
        self.definitions = definitions
        self._agent_runtime: AgentRuntimeManager | None = None
        self._team_manager: TeamManager | None = None
        self._storage_dir: str | None = None
        self._max_background_tasks = 4

    def bind_collaboration(
        self,
        *,
        agent_runtime: "AgentRuntimeManager",
        team_manager: "TeamManager",
        storage_dir: str,
        max_background_tasks: int,
    ) -> None:
        self._agent_runtime = agent_runtime
        self._team_manager = team_manager
        self._storage_dir = storage_dir
        self._max_background_tasks = max_background_tasks

    def cache_safe_params(self) -> dict[str, Any]:
        cache_policy = getattr(self.config, "cache_policy", None)
        try:
            system_blocks = self.prompt_blocks_provider() if self.prompt_blocks_provider else []
            tools = self.provider_tools_provider() if self.provider_tools_provider else None
            signature = build_cache_signature(
                provider=getattr(self.llm, "provider_name", None),
                model=getattr(self.llm, "model", None),
                system_blocks=compile_prompt_blocks(
                    system_blocks,
                    cache_policy=cache_policy,
                ).system_prompt_blocks,
                tools=tools,
                reasoning=self.reasoning,
                cache_policy=cache_policy,
            ).to_dict()
        except Exception:
            signature = {}
        return {
            "provider": getattr(self.llm, "provider_name", None),
            "model": getattr(self.llm, "model", None),
            "reasoning": self.reasoning,
            "cachePolicy": cache_policy.to_dict() if hasattr(cache_policy, "to_dict") else cache_policy,
            "signature": signature,
        }

    def __call__(self, request: "SubagentRequest") -> Any:
        from agent.BasicAgent import BasicAgent
        from core.llm import EasyLLM

        definition = self.definitions.get(request.agent_type)
        workspace_root = request.workspace_root or self.runtime.workspace_root
        allowed_roots = tuple(request.allowed_roots or (workspace_root,))

        config = self.config.model_copy(deep=True) if hasattr(self.config, "model_copy") else deepcopy(self.config)
        if config is not None:
            config.workspace_root = workspace_root
            config.allowed_roots = list(allowed_roots)
            subagent_cache_policy = str(getattr(config, "subagent_cache_policy", "inherit") or "inherit")
            normalized_policy = PromptCachePolicy.from_value(getattr(config, "cache_policy", None))
            if subagent_cache_policy == "read_only":
                normalized_policy.mode = "read_only"
                config.cache_policy = normalized_policy
            elif subagent_cache_policy in {"skip_write", "isolated"}:
                normalized_policy.mode = "skip_write"
                config.cache_policy = normalized_policy

        child_llm = self.llm
        requested_model = request.model or definition.model
        if requested_model and requested_model != getattr(self.llm, "model", None):
            child_llm = EasyLLM(
                provider=self.llm.provider_name,
                model=requested_model,
                api_key=getattr(self.llm, "resolve_api_key", None) or self.llm.api_key,
                base_url=getattr(self.llm, "resolve_base_url", None) or self.llm.base_url,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
                timeout=self.llm.timeout,
                **dict(getattr(self.llm, "kwargs", {}) or {}),
            )

        requested_mode = request.mode or definition.permission_mode
        approval_mode = _subagent_approval_mode(requested_mode)
        task_id = request.metadata.get("task_id") if isinstance(request.metadata, dict) else None
        team_id = None
        if request.team_name and self._team_manager is not None:
            try:
                team_id = self._team_manager.get_team(request.team_name).team_id
            except KeyError:
                team_id = None
        child_context = self.runtime.derive(
            agent_id=request.metadata.get("agent_id"),
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            worktree_path=request.worktree_path,
            worktree_branch=request.worktree_branch,
            execution_mode="plan" if requested_mode == "plan" else "execute",
            permission_mode=approval_mode.value if approval_mode is not None else None,
            current_task_id=task_id,
            metadata={
                "delegatedBy": self.parent_name,
                "parentAgentId": self.runtime.metadata.get("agentId"),
                "agentId": request.metadata.get("agent_id"),
                "outputFile": request.metadata.get("output_file"),
                "teamName": request.team_name,
                "teamId": team_id,
            },
        )

        selected_names = {
            name for name in self.tool_registry.get_tool_names()
            if (definition.tools is None or name in definition.tools)
            and name not in definition.disallowed_tools
        }
        # Overrides are applied after module assembly, so factories can also
        # replace/omit module-contributed tools without registration conflicts.
        excluded = [
            tool for tool in self.tool_registry.get_visible_tools()
            if tool.name not in selected_names or tool.name in self.child_tool_factories
        ]
        registry = clone_tool_registry_for_workspace(
            self.tool_registry, workspace_root=workspace_root,
            allowed_roots=allowed_roots, cwd=workspace_root, runtime=child_context,
            excluded_tools=(
                *excluded, *self.parent_bound_tools,
                *(self.plan.get_tools() if self.plan else ()),
            ),
        )
        child = BasicAgent(
            name=request.name or request.description or "Subagent",
            llm=child_llm,
            system_prompt="\n\n".join(
                part
                for part in (self.system_prompt, definition.system_prompt)
                if part
            ) or None,
            description=request.description,
            config=config,
            runtime=child_context,
        )
        child.reasoning = deepcopy(self.reasoning)
        child_permission_engine = (
            self.child_permission_factory(self.permission_engine, child_context)
            if self.permission_engine is not None else None
        )
        if self.permission_engine is not None and not isinstance(child_permission_engine, BasePermissionEngine):
            raise TypeError("Child permission factory must return BasePermissionEngine")
        if child_permission_engine is None and approval_mode is not None:
            child_permission_engine = RuleBasedPermissionEngine(
                approval_mode=approval_mode, workspace_root=workspace_root, load_policies=False,
            )
        if child_permission_engine is not None and child_permission_engine is self.permission_engine:
            raise ValueError("Child permission factory must not rebind the parent engine")
        if child_permission_engine is not None:
            child.with_permissions(child_permission_engine)
        if self.callback_manager is not None:
            child.with_callbacks(self.callback_manager)
        child.with_tool(registry)
        if self.plan is not None:
            child_plan = self.plan.create_child()
            if child_plan is self.plan:
                raise ValueError("Plan.create_child must not return the parent instance")
            if child_plan is not None:
                child.with_plan(child_plan)
        if self.skill_directories:
            child.with_skill(*self.skill_directories)
        if self.task_service is not None:
            child.with_task_service(self.task_service)
        if self.observability is not None and getattr(child, "observability", None) is None:
            child_observability = self.observability.create_child(
                parent_invoke_id=request.metadata.get("parent_agent_invoke_id"),
            )
            if child_observability is self.observability:
                raise ValueError("Observability.create_child must return an independent binding")
            child.with_observability(manager=child_observability)
        if bool(getattr(config, "subagent_inherit_cache_safe_params", True)):
            child_context.metadata.setdefault("cacheSafeParams", self.cache_safe_params())

        if (
            request.team_name is not None
            and self._agent_runtime is not None
            and self._team_manager is not None
        ):
            from multiagent.default.module import ClaudeCodeMultiAgent

            child.with_multi_agent(
                ClaudeCodeMultiAgent(
                    workspace_root=workspace_root,
                    storage_dir=self._storage_dir,
                    max_background_tasks=self._max_background_tasks,
                    agent_factory=self,
                    agent_runtime=self._agent_runtime,
                    team_manager=self._team_manager,
                    agent_name=child.name,
                    task_service=self.task_service,
                    definitions=self.definitions,
                )
            )
        apply_child_tool_factories(
            registry, self.tool_registry, child_context,
            {name: factory for name, factory in self.child_tool_factories.items()
             if name in selected_names},
        )
        # Modules contribute fresh child-bound tools. Preserve the parent's
        # explicit security descriptor even for those reconstructed capabilities.
        for name in registry.get_tool_names():
            if self.tool_registry.has_tool(name):
                spec = registry.get_tool_spec(name)
                registry.register_tool(
                    registry.get_tool(name),
                    visibility=registry.get_tool_visibility(name) or "resident",
                    expose_in_deferred=spec.expose_in_deferred,
                    permission_descriptor=deepcopy(self.tool_registry.get_permission_descriptor(name)),
                )
        if definition.tools is not None:
            registry.unregister_tools([name for name in registry.get_tool_names() if name not in definition.tools])
        registry.unregister_tools([name for name in definition.disallowed_tools if registry.has_tool(name)])
        return child


__all__ = ["DefaultSubagentFactory"]
