"""Tool wrapper for creating multi-agent teams."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from multiagent.default.teams import TeamManager
from runtime import AgentRuntime

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.display_utils import format_structured_display


class TeamCreateParams(BaseModel):
    team_name: str = Field(description="Name for the new team to create")
    description: str = Field(default="", description="Team description")
    agent_type: Optional[str] = Field(
        default=None,
        description="Type or role of the team lead",
    )


TEAM_CREATE_PROMPT = """# TeamCreate

## When to Use

Use this tool when the user explicitly asks for a team or when a complex task benefits from several agents working in parallel. Do not create a team for work one agent can complete directly.

## Choosing Agent Types for Teammates

When spawning teammates with Agent, select a subagent type whose tools match the task. Read-only agents should receive research or planning work; agents with editing and shell tools can receive implementation work.

## Team Workflow

1. Create a team with TeamCreate.
2. Create shared tasks with the Task tools when task tracking is installed.
3. Spawn teammates with Agent using `team_name` and a short `name`.
4. Assign task ownership with TaskUpdate.
5. Coordinate through SendMessage. Ordinary assistant text is not delivered to teammates.
6. Teammates should report results and update task state before becoming idle.
7. Gracefully stop active teammates before deleting the team.

Messages from teammates are delivered automatically. Do not manually poll the mailbox for normal coordination. Refer to teammates by name rather than opaque IDs."""


class TeamCreateTool(Tool):
    def __init__(
        self,
        *,
        team_manager: TeamManager,
        creator_name: str | None = None,
        runtime: AgentRuntime | None = None,
    ):
        self.team_manager = team_manager
        self.creator_name = creator_name
        self.runtime = runtime
        super().__init__(
            name="TeamCreate",
            description="Creates an explicit agent team.",
            parameters=TeamCreateParams,
            guidance=(
                "Create the team before launching agents with team_name; this does not launch members."
            ),
            prompt=TEAM_CREATE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
            side_effect_level="medium",
            resource_scope=["runtime", "team", "task"],
        )

    def run(self, parameters: dict) -> ToolResult:
        metadata: dict[str, Any] = {}
        current_task_id = getattr(self.runtime, "current_task_id", None)
        if current_task_id and "taskId" not in metadata:
            metadata["taskId"] = current_task_id
        creator_name = self.creator_name
        if creator_name and "createdBy" not in metadata:
            metadata["createdBy"] = creator_name
        if parameters.get("agent_type"):
            metadata["leadAgentType"] = parameters["agent_type"]
        try:
            team = self.team_manager.create_team(
                name=parameters["team_name"],
                description=str(parameters.get("description") or ""),
                metadata=metadata,
            )
            if self.runtime is not None:
                self.runtime.metadata["teamName"] = team.name
                self.runtime.metadata["teamId"] = team.team_id
        except Exception as exc:
            return ToolResult.error(
                f"创建团队失败: {exc}",
                error_type="team_create_failed",
                metadata={"team_name": parameters.get("team_name")},
            )
        payload = team.to_dict()
        return ToolResult.success(
            content=f"已创建团队 {team.name}",
            display_text=format_structured_display(
                f"已创建团队 {team.name}",
                payload,
            ),
            structured_data=payload,
            metadata=payload,
        )


def register_team_create_tool(
    registry: ToolRegistry,
    *,
    team_manager: TeamManager,
    creator_name: str | None = None,
    runtime: AgentRuntime | None = None,
    expose_in_deferred: bool | None = True,
) -> TeamCreateTool:
    tool = TeamCreateTool(
        team_manager=team_manager,
        creator_name=creator_name,
        runtime=runtime,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = ["TeamCreateTool", "register_team_create_tool"]
