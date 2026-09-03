"""Optional multi-agent collaboration module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

from metamessage import BaseMetaMessageManager, MetaMessage, MetaMessageLifecycle
from Tool.ToolRegistry import ToolRegistry

from .agents import AgentRuntimeManager
from .context import ExecutionContext
from .teams import TeamManager


class BaseMultiAgentRuntime(ABC):
    @abstractmethod
    def install(self, agent: Any) -> "BaseMultiAgentRuntime":
        raise NotImplementedError

    @abstractmethod
    def sync_mailbox(
        self,
        *,
        execution_context: ExecutionContext,
        metamessage_manager: BaseMetaMessageManager,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> dict[str, Any]:
        raise NotImplementedError


class MultiAgentRuntime(BaseMultiAgentRuntime):
    """Facade owning subagent, team, mailbox, tools, restore, and shutdown."""

    def __init__(
        self,
        *,
        workspace_root: str | None = None,
        storage_dir: str | None = None,
        max_background_tasks: int = 4,
        agent_runtime: AgentRuntimeManager | None = None,
        team_manager: TeamManager | None = None,
    ) -> None:
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self.storage_dir = os.path.abspath(
            storage_dir or os.path.join(self.workspace_root, ".easyagent-agents")
        )
        self.max_background_tasks = max(1, int(max_background_tasks))
        self.agent_runtime = agent_runtime
        self.team_manager = team_manager
        self._agent: Any = None

    def install(self, agent: Any) -> "MultiAgentRuntime":
        from Tool.builtin import (
            register_agent_runtime_tools,
            register_agent_tool,
            register_mailbox_tools,
            register_send_message_tool,
            register_team_create_tool,
            register_team_delete_tool,
        )

        registry = getattr(agent, "tool_registry", None)
        if not isinstance(registry, ToolRegistry):
            raise TypeError("MultiAgentRuntime requires an installed ToolRegistry")
        self._agent = agent
        allowed_roots = tuple(agent.execution_context.allowed_roots)
        agent_tool = register_agent_tool(
            registry,
            parent_agent=agent,
            workspace_root=self.workspace_root,
            allowed_roots=allowed_roots,
            storage_dir=self.storage_dir,
            max_background_tasks=self.max_background_tasks,
            agent_runtime=self.agent_runtime,
            worktree_manager=getattr(agent, "worktree_manager", None),
        )
        self.agent_runtime = agent_tool.agent_runtime
        if self.team_manager is None:
            self.team_manager = TeamManager(agent_runtime=self.agent_runtime)
        self.agent_runtime.bind_team_manager(self.team_manager)
        register_agent_runtime_tools(registry, agent_runtime=self.agent_runtime, parent_agent=agent)
        register_send_message_tool(registry, agent_runtime=self.agent_runtime, parent_agent=agent)
        register_mailbox_tools(registry, agent_runtime=self.agent_runtime, parent_agent=agent)
        register_team_create_tool(registry, team_manager=self.team_manager, parent_agent=agent)
        register_team_delete_tool(registry, team_manager=self.team_manager)
        return self

    def sync_mailbox(
        self,
        *,
        execution_context: ExecutionContext,
        metamessage_manager: BaseMetaMessageManager,
    ) -> None:
        if self.agent_runtime is None:
            return
        agent_id = str(execution_context.metadata.get("agentId") or "").strip()
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
            metamessage_manager.emit(
                MetaMessage(
                    name=f"mailbox:{message.message_id}",
                    content=(
                        "A runtime mailbox message was delivered to this agent. Treat it as current "
                        "collaboration input and adjust the task when it changes goals or constraints.\n"
                        f"Sender: {message.sender_id or 'unknown'}\n"
                        f"Message ID: {message.message_id}\n"
                        f"Content: {message.content}"
                    ),
                    lifecycle=MetaMessageLifecycle.PERMANENT,
                    dedup_key=f"mailbox:{message.message_id}",
                    metadata={
                        "source": "mailbox",
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
            "agentRuntime": self.agent_runtime.export_state() if self.agent_runtime is not None else None,
            "teams": self.team_manager.export_state() if self.team_manager is not None else None,
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

        def merge(component: str, component_report: dict[str, Any]) -> None:
            report[component] = component_report
            report["restoredItems"].extend(
                f"{component}:{item}"
                for item in list(component_report.get("restoredItems") or [])
            )
            report["degradedItems"].extend(
                f"{component}:{item}"
                for item in list(component_report.get("degradedItems") or [])
            )
            report["missingItems"].extend(
                f"{component}:{item}"
                for item in list(component_report.get("missingItems") or [])
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

        if self.agent_runtime is not None and payload.get("agentRuntime"):
            runtime_report = self.agent_runtime.restore_state(payload.get("agentRuntime"))
            merge("agentRuntime", runtime_report)
        if self.team_manager is not None and payload.get("teams"):
            merge("teams", self.team_manager.restore_state(payload.get("teams")))
        return report

    def close(self) -> dict[str, Any]:
        if self.agent_runtime is None:
            return {"status": "closed", "issues": []}
        return self.agent_runtime.close()


__all__ = ["BaseMultiAgentRuntime", "MultiAgentRuntime"]
