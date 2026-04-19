"""Tool wrapper for creating multi-agent teams."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from runtime import TeamManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry


class TeamCreateParams(BaseModel):
    name: str = Field(description="团队名称")
    description: str = Field(default="", description="团队描述")
    member_agent_ids: list[str] = Field(default_factory=list, description="初始成员 agent ID 列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class TeamCreateTool(Tool):
    def __init__(self, *, team_manager: TeamManager):
        self.team_manager = team_manager
        super().__init__(
            name="TeamCreate",
            description="创建一个显式的 agent 团队。",
            parameters=TeamCreateParams,
            guidance="先创建 team，再让子 agent 用相同 team_name 加入该团队。",
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            team = self.team_manager.create_team(**parameters)
        except Exception as exc:
            return ToolResult.error(
                f"创建团队失败: {exc}",
                error_type="team_create_failed",
                metadata={"team_name": parameters.get("name")},
            )
        payload = team.to_dict()
        return ToolResult.success(
            content=f"已创建团队 {team.name}",
            structured_data=payload,
            metadata=payload,
        )


def register_team_create_tool(
    registry: ToolRegistry,
    *,
    team_manager: TeamManager,
) -> TeamCreateTool:
    tool = TeamCreateTool(team_manager=team_manager)
    registry.register_tool(tool)
    return tool


__all__ = ["TeamCreateTool", "register_team_create_tool"]
