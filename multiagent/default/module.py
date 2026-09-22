"""Built-in Claude Code-style multi-agent module."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from multiagent.base import BaseMultiAgent
from prompt import PromptBlock
from runtime import AgentRuntime, RuntimeEventBus
from runtime.metamessage import MetaMessage, MetaMessageLifecycle, MetaMessageManager
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from Tool.runtime import WorktreeManager

from .agents import AgentRuntimeManager
from .definitions import AgentDefinitionRegistry, build_default_agent_definitions
from .prompts import CLAUDE_MULTI_AGENT_PROMPT
from .subagents import SubagentRequest
from .teams import TeamManager
from .tools import (
    AgentGetTool,
    AgentListTool,
    AgentStopTool,
    AgentTool,
    AgentWaitTool,
    SendMessageTool,
    TeamCreateTool,
    TeamDeleteTool,
)

if TYPE_CHECKING:
    from prompt import PromptBuildContext


class ClaudeCodeMultiAgent(BaseMultiAgent):
    """Default collaboration harness inspired by Claude Code.

    Subagents, teams, mailboxes, background execution, persistence and
    worktree isolation are implementation details here, not framework-level
    requirements imposed on custom multi-agent modules.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[[SubagentRequest], Any],
        workspace_root: str | None = None,
        storage_dir: str | None = None,
        max_background_tasks: int = 4,
        agent_runtime: AgentRuntimeManager | None = None,
        team_manager: TeamManager | None = None,
        worktree_manager: WorktreeManager | None = None,
        agent_name: str | None = None,
        task_service: Any = None,
        observability: Any = None,
        cache_safe_params_provider: Callable[[], dict[str, Any]] | None = None,
        definitions: AgentDefinitionRegistry | None = None,
    ) -> None:
        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable")
        self.agent_factory = agent_factory
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self.storage_dir = os.path.abspath(
            storage_dir or os.path.join(self.workspace_root, ".easyagent-agents")
        )
        self.max_background_tasks = max(1, int(max_background_tasks))
        self.agent_runtime = agent_runtime or AgentRuntimeManager(
            agent_factory=agent_factory,
            storage_dir=self.storage_dir,
            max_background_tasks=self.max_background_tasks,
        )
        self.team_manager = team_manager or TeamManager(agent_runtime=self.agent_runtime)
        self.agent_runtime.bind_team_manager(self.team_manager)
        self.worktree_manager = worktree_manager
        self.agent_name = agent_name
        self.task_service = task_service
        self.observability = observability
        self.cache_safe_params_provider = cache_safe_params_provider
        self.definitions = definitions or build_default_agent_definitions()
        self._tool_registry: ToolRegistry | None = None
        self._runtime: AgentRuntime | None = None
        self._tools: tuple[Tool, ...] = ()

        bind_collaboration = getattr(agent_factory, "bind_collaboration", None)
        if callable(bind_collaboration):
            bind_collaboration(
                agent_runtime=self.agent_runtime,
                team_manager=self.team_manager,
                storage_dir=self.storage_dir,
                max_background_tasks=self.max_background_tasks,
            )

    def bind(
        self,
        *,
        tool_registry: ToolRegistry,
        runtime: AgentRuntime,
    ) -> None:
        self._tool_registry = tool_registry
        self._runtime = runtime
        self._tools = self._build_tools()

    def _build_tools(self) -> tuple[Tool, ...]:
        if self._tool_registry is None or self._runtime is None:
            raise RuntimeError("ClaudeCodeMultiAgent must be bound before tools are requested")
        common = {
            "agent_runtime": self.agent_runtime,
            "task_service": self.task_service,
        }
        return (
            AgentTool(
                agent_factory=self.agent_factory,
                runtime=self._runtime,
                tool_registry=self._tool_registry,
                agent_name=self.agent_name,
                task_service=self.task_service,
                observability=self.observability,
                cache_safe_params_provider=self.cache_safe_params_provider,
                agent_runtime=self.agent_runtime,
                worktree_manager=self.worktree_manager,
                workspace_root=self.workspace_root,
                allowed_roots=self._runtime.allowed_roots,
                storage_dir=self.storage_dir,
                max_background_tasks=self.max_background_tasks,
            ),
            AgentGetTool(**common),
            AgentListTool(**common),
            AgentWaitTool(**common),
            AgentStopTool(**common),
            SendMessageTool(
                agent_runtime=self.agent_runtime,
                sender_name=self.agent_name,
                runtime=self._runtime,
            ),
            TeamCreateTool(
                team_manager=self.team_manager,
                creator_name=self.agent_name,
                runtime=self._runtime,
            ),
            TeamDeleteTool(
                team_manager=self.team_manager,
                runtime=self._runtime,
            ),
        )

    def get_tools(self) -> Sequence[Tool]:
        if not self._tools:
            raise RuntimeError("ClaudeCodeMultiAgent is not bound")
        return self._tools

    def get_prompt_blocks(self, context: "PromptBuildContext") -> Sequence[PromptBlock]:
        agent_listing = "\n".join(
            definition.format_listing()
            for definition in self.definitions.list()
        )
        return (
            PromptBlock(
                "multiagent.claude_code",
                (
                    CLAUDE_MULTI_AGENT_PROMPT
                    + "\n\nAvailable agent types and the tools they have access to:\n"
                    + agent_listing
                ),
                placement="system_reminder",
                order=170,
                metadata={"cache_partition": "session", "cacheable": True},
            ),
        )

    def before_context_build(self) -> None:
        if self._runtime is None or self._runtime.metamessages is None:
            return
        agent_id = str(self._runtime.metadata.get("agentId") or "").strip()
        if not agent_id:
            return
        try:
            messages = self.agent_runtime.read_mailbox(
                agent_id,
                include_consumed=False,
                include_expired=False,
                mark_delivered=True,
            )
        except KeyError:
            return
        for message in messages:
            self._runtime.metamessages.emit(
                MetaMessage(
                    name=f"teammate:{message.message_id}",
                    content=(
                        f'<teammate-message teammate_id="{message.sender_id or "unknown"}">\n'
                        f"{message.content}\n"
                        "</teammate-message>"
                    ),
                    lifecycle=MetaMessageLifecycle.PERMANENT,
                    dedup_key=f"teammate:{message.message_id}",
                    metadata={
                        "source": "multiagent",
                        "mailboxMessageId": message.message_id,
                        "senderId": message.sender_id,
                    },
                )
            )

    def export_state(self) -> dict[str, Any]:
        return {
            "workspaceRoot": self.workspace_root,
            "storageDir": self.storage_dir,
            "maxBackgroundTasks": self.max_background_tasks,
            "agentRuntime": self.agent_runtime.export_state(),
            "teams": self.team_manager.export_state(),
        }

    def restore_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(state or {})
        report: dict[str, Any] = {
            "status": "restored",
            "restoredItems": [],
            "degradedItems": [],
            "missingItems": [],
            "metadata": {},
            "issues": [],
        }

        def merge(component: str, component_report: Mapping[str, Any]) -> None:
            for field in ("restoredItems", "degradedItems", "missingItems"):
                report[field].extend(
                    f"{component}:{item}" for item in list(component_report.get(field) or [])
                )
            report["issues"].extend(
                {**dict(issue), "component": component}
                for issue in list(component_report.get("issues") or [])
            )
            report["metadata"][component] = dict(component_report.get("metadata") or {})
            status = str(component_report.get("status") or "restored")
            if status == "failed":
                report["status"] = "failed"
            elif status == "degraded" and report["status"] == "restored":
                report["status"] = "degraded"

        if payload.get("teams"):
            merge("teams", self.team_manager.restore_state(payload.get("teams")))
        if payload.get("agentRuntime"):
            merge("agentRuntime", self.agent_runtime.restore_state(payload.get("agentRuntime")))
        return report

    def close(self) -> dict[str, Any]:
        return dict(self.agent_runtime.close())


__all__ = ["ClaudeCodeMultiAgent"]
