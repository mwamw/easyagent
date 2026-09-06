"""Tool wrapper for deleting multi-agent teams."""

from __future__ import annotations

from pydantic import BaseModel, Field

from runtime import TeamManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


class TeamDeleteParams(BaseModel):
    team_id: str = Field(description="ID or name of the team to delete")


TEAM_DELETE_PROMPT = """Deletes a team handle.
- Supply team_id or the team name.
- This removes the team record; it does not stop member agents or delete their worktrees and task records.
- Stop or finish members as appropriate before removing the team.
- Do not treat successful deletion as proof that background work has stopped."""


class TeamDeleteTool(Tool):
    def __init__(self, *, team_manager: TeamManager):
        self.team_manager = team_manager
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
        identifier = parameters["team_id"]
        try:
            team = self.team_manager.delete_team(identifier)
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
    expose_in_deferred: bool | None = True,
) -> TeamDeleteTool:
    tool = TeamDeleteTool(team_manager=team_manager)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = ["TeamDeleteTool", "register_team_delete_tool"]
