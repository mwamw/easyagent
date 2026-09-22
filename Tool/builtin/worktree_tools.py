"""Claude-style git worktree tools."""

from __future__ import annotations

from typing import Optional

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeEnterWorktreeInput, ClaudeExitWorktreeInput
from worktree import BaseWorktreeManager


ENTER_WORKTREE_PROMPT = """Creates and enters an isolated git worktree.
- Use when changes should be isolated from the main workspace.
- Each manager allows one active worktree at a time.
- Run subsequent work in the returned worktree path.
- This does not automatically commit, push, or clean up changes.
- Use ExitWorktree explicitly when finished: keep preserves the result; remove cleans up the worktree."""

EXIT_WORKTREE_PROMPT = """Exits the active git worktree.
- action="keep" leaves its directory and branch available for later inspection.
- action="remove" removes the worktree.
- discard_changes=true applies only to remove and may discard uncommitted changes and ahead commits.
- Confirm destructive cleanup with the user when it is not already authorized. Keep the worktree if uncertain whether its changes should be preserved."""


class EnterWorktreeTool(Tool):
    def __init__(self, *, worktree_manager: BaseWorktreeManager):
        self.worktree_manager = worktree_manager
        super().__init__(
            name="EnterWorktree",
            description="Creates and enters an isolated git worktree.",
            parameters=ClaudeEnterWorktreeInput,
            guidance="Create isolation before editing. Exit the current worktree before entering another.",
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
    def __init__(self, *, worktree_manager: BaseWorktreeManager):
        self.worktree_manager = worktree_manager
        super().__init__(
            name="ExitWorktree",
            description="Exits the active worktree, optionally removing it.",
            parameters=ClaudeExitWorktreeInput,
            guidance="Use keep to preserve work; remove only when cleanup is intended.",
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
    worktree_manager: BaseWorktreeManager,
    expose_in_deferred: bool | None = True,
) -> EnterWorktreeTool:
    tool = EnterWorktreeTool(worktree_manager=worktree_manager)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_exit_worktree_tool(
    registry: ToolRegistry,
    *,
    worktree_manager: BaseWorktreeManager,
    expose_in_deferred: bool | None = True,
) -> ExitWorktreeTool:
    tool = ExitWorktreeTool(worktree_manager=worktree_manager)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_worktree_tools(
    registry: ToolRegistry,
    *,
    worktree_manager: BaseWorktreeManager,
    expose_in_deferred: bool | None = True,
) -> tuple[EnterWorktreeTool, ExitWorktreeTool]:
    enter_tool = register_enter_worktree_tool(registry, worktree_manager=worktree_manager, expose_in_deferred=expose_in_deferred)
    exit_tool = register_exit_worktree_tool(registry, worktree_manager=worktree_manager, expose_in_deferred=expose_in_deferred)
    return enter_tool, exit_tool


__all__ = [
    "EnterWorktreeTool",
    "ExitWorktreeTool",
    "register_enter_worktree_tool",
    "register_exit_worktree_tool",
    "register_worktree_tools",
]
