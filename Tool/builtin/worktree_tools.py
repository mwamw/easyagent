"""Claude-style git worktree tools."""

from __future__ import annotations

from typing import Optional

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeEnterWorktreeInput, ClaudeExitWorktreeInput
from ..runtime import WorktreeManager


ENTER_WORKTREE_PROMPT = """创建并进入一个隔离的 git worktree。

何时使用：
- 你准备做代码修改，但不希望污染当前主工作区。
- 你需要给某个子任务或子 agent 一个独立分支和独立文件树。
- 你需要在保留主仓库干净状态的同时，安全地试验修改。

重要语义：
- 进入后会记录当前活动 worktree；后续应通过 `ExitWorktree` 明确退出或移除。
- 一个 worktree manager 同一时刻只允许一个 active session，避免混乱切换。
- worktree 只是隔离工作区，不自动提交、不自动推送，也不自动清理。

最佳实践：
- 在真正修改代码之前调用，而不是修改之后再切。
- 如果这个 worktree 只为一次临时任务服务，结束时通常应搭配 `ExitWorktree(action=\"remove\")`。
- 如果你还需要保留结果供后续查看，先用 `action=\"keep\"` 退出，再决定后续如何处理。"""

EXIT_WORKTREE_PROMPT = """退出当前活动 worktree，或直接将其移除。

何时使用：
- 当前 worktree 任务已经完成，需要回到原工作区。
- 当前隔离分支只是临时产物，准备直接清理掉。

动作语义：
- `action=\"keep\"`：只退出当前 worktree，保留目录和分支，适合后续还要回来检查。
- `action=\"remove\"`：移除当前 worktree。
- `discard_changes=true` 只对 `action=\"remove\"` 有意义，会强制丢弃未提交改动和 ahead commits。

注意：
- 如果你还不确定这个 worktree 是否需要保留，不要急着 `remove`。
- `remove + discard_changes=true` 属于高风险清理操作，只在你确定这些改动可以丢弃时使用。"""


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
