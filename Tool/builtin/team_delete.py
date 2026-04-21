"""Tool wrapper for deleting multi-agent teams."""

from __future__ import annotations

from pydantic import BaseModel, Field

from runtime import TeamManager

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_structured_display


class TeamDeleteParams(BaseModel):
    team_id: str = Field(description="要删除的团队 ID，也支持团队名称")


TEAM_DELETE_PROMPT = """删除一个 team 句柄。

何时使用：
- 当某个协作团队已经完成使命，不再需要作为广播或管理单元存在时。
- 当你需要清理临时 team，避免后续消息继续广播给这组成员时。

重要语义：
- 删除 team 不会删除成员 agent，也不会停止它们正在进行的任务。
- 删除 team 也不会删除相关 task、mailbox 历史或 outputFile。
- 如果你只是暂时不想广播消息，而不是彻底解散团队，不要急着删除。"""


class TeamDeleteTool(Tool):
    def __init__(self, *, team_manager: TeamManager):
        self.team_manager = team_manager
        super().__init__(
            name="TeamDelete",
            description="删除一个 agent 团队。",
            parameters=TeamDeleteParams,
            guidance="删除团队句柄不会停止成员 agent，也不会删除已有 mailbox / task / outputFile 状态。",
            prompt=TEAM_DELETE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["agent", "team", "collaboration"],
            risk_categories=["side_effect"],
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
) -> TeamDeleteTool:
    tool = TeamDeleteTool(team_manager=team_manager)
    registry.register_tool(tool)
    return tool


__all__ = ["TeamDeleteTool", "register_team_delete_tool"]
