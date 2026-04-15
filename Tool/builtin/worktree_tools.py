"""Claude-style git worktree tools."""

from __future__ import annotations

from typing import Optional

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeEnterWorktreeInput, ClaudeExitWorktreeInput
from ..runtime import WorktreeManager


ENTER_WORKTREE_PROMPT = """用于创建并进入一个隔离的 git worktree。
- 适合需要隔离代码修改、避免污染主工作区时使用。
- 进入后会记录当前活动 worktree，后续可通过 ExitWorktree 退出。"""

EXIT_WORKTREE_PROMPT = """用于退出当前活动 worktree，或直接将其移除。
- `action=keep` 只退出，不删除 worktree。
- `action=remove` 会移除当前 worktree；若 `discard_changes=true`，会强制移除未提交改动。"""


class EnterWorktreeTool(Tool):
    def __init__(self, *, worktree_manager: WorktreeManager):
        self.worktree_manager = worktree_manager
        super().__init__(
            name="EnterWorktree",
            description="创建并进入一个隔离 git worktree。",
            parameters=ClaudeEnterWorktreeInput,
            guidance="适合在代码修改前创建隔离工作区。若已存在活动 worktree，应先退出。",
            prompt=ENTER_WORKTREE_PROMPT,
            read_only=False,
            destructive=True,
            supports_parallel=False,
            source="builtin",
            tags=["git", "worktree", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        name = parameters.get("name")
        try:
            session = self.worktree_manager.enter_worktree(name=name)
        except Exception as exc:
            return ToolResult.error(
                f"进入 worktree 失败: {exc}",
                error_type="worktree_enter_failed",
                metadata={"name": name},
            )

        structured_data = {
            "worktreePath": session.worktree.path,
            "worktreeBranch": session.worktree.branch,
            "message": f"已进入隔离 worktree: {session.worktree.path}",
        }
        return ToolResult.success(
            structured_data["message"],
            structured_data=structured_data,
            metadata=structured_data,
        )


class ExitWorktreeTool(Tool):
    def __init__(self, *, worktree_manager: WorktreeManager):
        self.worktree_manager = worktree_manager
        super().__init__(
            name="ExitWorktree",
            description="退出当前活动 worktree，或直接将其移除。",
            parameters=ClaudeExitWorktreeInput,
            guidance="需要保留 worktree 时用 action=keep；确认不再需要时再用 action=remove。",
            prompt=EXIT_WORKTREE_PROMPT,
            read_only=False,
            destructive=True,
            supports_parallel=False,
            source="builtin",
            tags=["git", "worktree", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        action = str(parameters.get("action", "")).strip()
        discard_changes = bool(parameters.get("discard_changes", False))
        try:
            result = self.worktree_manager.exit_worktree(action, discard_changes=discard_changes)
        except Exception as exc:
            return ToolResult.error(
                f"退出 worktree 失败: {exc}",
                error_type="worktree_exit_failed",
                metadata={"action": action, "discard_changes": discard_changes},
            )

        return ToolResult.success(
            str(result.get("message", "")),
            structured_data=result,
            metadata=result,
        )


def register_enter_worktree_tool(
    registry: ToolRegistry,
    *,
    worktree_manager: WorktreeManager,
) -> EnterWorktreeTool:
    tool = EnterWorktreeTool(worktree_manager=worktree_manager)
    registry.register_tool(tool)
    return tool


def register_exit_worktree_tool(
    registry: ToolRegistry,
    *,
    worktree_manager: WorktreeManager,
) -> ExitWorktreeTool:
    tool = ExitWorktreeTool(worktree_manager=worktree_manager)
    registry.register_tool(tool)
    return tool


def register_worktree_tools(
    registry: ToolRegistry,
    *,
    worktree_manager: WorktreeManager,
) -> tuple[EnterWorktreeTool, ExitWorktreeTool]:
    enter_tool = register_enter_worktree_tool(registry, worktree_manager=worktree_manager)
    exit_tool = register_exit_worktree_tool(registry, worktree_manager=worktree_manager)
    return enter_tool, exit_tool


__all__ = [
    "EnterWorktreeTool",
    "ExitWorktreeTool",
    "register_enter_worktree_tool",
    "register_exit_worktree_tool",
    "register_worktree_tools",
]
