"""Claude-style background task stop tool."""

from __future__ import annotations

from typing import Iterable, Optional

from ..BaseTool import ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeTaskStopInput
from ..runtime import ProcessManager
from .bash_tool import DEFAULT_BASH_OUTPUT_CHARS, _ShellToolBase, _format_background_snapshot, _snapshot_to_dict


class TaskStopTool(_ShellToolBase):
    """Stop a background task started by Bash."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        shell: str = "bash",
        command_timeout_ms: int = 120000,
        max_background_tasks: int = 8,
        max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
        process_manager: Optional[ProcessManager] = None,
    ):
        super().__init__(
            name="TaskStop",
            description="停止 Bash 后台任务。",
            parameters=ClaudeTaskStopInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            shell=shell,
            command_timeout_ms=command_timeout_ms,
            max_background_tasks=max_background_tasks,
            max_output_chars=max_output_chars,
            process_manager=process_manager,
            guidance="用于终止已启动的后台任务。优先传 task_id；shell_id 会作为兼容别名处理。",
            prompt="用于终止通过 Bash 启动的后台任务。",
            read_only=False,
            destructive=True,
            supports_parallel=False,
            tags=["shell", "background", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        task_id = str(parameters.get("task_id") or parameters.get("shell_id") or "").strip()
        if not task_id:
            return self._tool_error(
                "task_id 或 shell_id 不能为空。",
                error_type="invalid_parameters",
                structured_data={"reason": "empty_task_id"},
            )

        try:
            snapshot = self.process_manager.stop(task_id)
        except KeyError as exc:
            return self._tool_error(
                str(exc),
                error_type="task_not_found",
                metadata={"task_id": task_id},
                structured_data={"reason": "task_not_found", "task_id": task_id},
            )

        display_text, truncated = _format_background_snapshot(snapshot, max_output_chars=self.max_output_chars)
        return ToolResult.success(
            display_text,
            structured_data={
                **_snapshot_to_dict(snapshot),
                "truncated": truncated,
            },
            metadata={
                "task_id": snapshot.task_id,
                "status": snapshot.status,
                "return_code": snapshot.return_code,
                "truncated": truncated,
            },
        )


def register_task_stop_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    shell: str = "bash",
    command_timeout_ms: int = 120000,
    max_background_tasks: int = 8,
    max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
    process_manager: Optional[ProcessManager] = None,
    expose_in_deferred: bool | None = True,
) -> TaskStopTool:
    tool = TaskStopTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=process_manager,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = [
    "TaskStopTool",
    "register_task_stop_tool",
]
