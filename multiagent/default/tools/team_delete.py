"""Tool wrapper for deleting multi-agent teams."""

from __future__ import annotations

from pydantic import BaseModel

from multiagent.default.teams import TeamManager
from runtime import AgentRuntime

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.display_utils import format_structured_display


class TeamDeleteParams(BaseModel):
    pass


TEAM_DELETE_PROMPT = """# TeamDelete

Remove the current team when collaborative work is complete.

This operation removes the team record and clears the current team context. It does not silently terminate running agents or delete their worktrees.

IMPORTANT: TeamDelete fails while the team still has active members. Gracefully stop teammates first, then call TeamDelete after they have shut down."""


class TeamDeleteTool(Tool):
    def __init__(
        self,
        *,
        team_manager: TeamManager,
        runtime: AgentRuntime | None = None,
    ):
        self.team_manager = team_manager
        self.runtime = runtime
        super().__init__(
            name="TeamDelete",
            description="Deletes an agent team.",
            parameters=TeamDeleteParams,
            guidance="Deleting the team does not stop agents or delete their mailboxes, tasks, or output files.",
            prompt=TEAM_DELETE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
            side_effect_level="medium",
            resource_scope=["runtime", "team"],
        )

    def run(self, parameters: dict) -> ToolResult:
        metadata = dict(getattr(self.runtime, "metadata", {}) or {})
        identifier = str(metadata.get("teamName") or metadata.get("teamId") or "").strip()
        if not identifier:
            return ToolResult.error(
                "No active team context.",
                error_type="team_delete_failed",
            )
        try:
            team = self.team_manager.delete_team(identifier)
            if self.runtime is not None:
                self.runtime.metadata.pop("teamName", None)
                self.runtime.metadata.pop("teamId", None)
        except Exception as exc:
            return ToolResult.error(
                f"删除团队失败: {exc}",
                error_type="team_delete_failed",
                metadata={"team_id": identifier},
            )
        payload = team.to_dict()
        return ToolResult.success(
            content=f"已删除团队 {team.name}",
            display_text=format_structured_display(
                f"已删除团队 {team.name}",
                payload,
            ),
            structured_data=payload,
            metadata=payload,
        )


def register_team_delete_tool(
    registry: ToolRegistry,
    *,
    team_manager: TeamManager,
    runtime: AgentRuntime | None = None,
    expose_in_deferred: bool | None = True,
) -> TeamDeleteTool:
    tool = TeamDeleteTool(
        team_manager=team_manager,
        runtime=runtime,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = ["TeamDeleteTool", "register_team_delete_tool"]
