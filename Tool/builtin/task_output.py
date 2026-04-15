"""Claude-style background task output tool."""

from __future__ import annotations

from typing import Iterable, Optional

from ..BaseTool import ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeTaskOutputInput
from ..runtime import ProcessManager
from .bash_tool import DEFAULT_BASH_OUTPUT_CHARS, _ShellToolBase, _format_background_snapshot, _snapshot_to_dict


class TaskOutputTool(_ShellToolBase):
    """Read output from a background task started by Bash."""

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
            name="TaskOutput",
            description="读取 Bash 后台任务的当前输出，可选择阻塞等待任务结束。",
            parameters=ClaudeTaskOutputInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            shell=shell,
            command_timeout_ms=command_timeout_ms,
            max_background_tasks=max_background_tasks,
            max_output_chars=max_output_chars,
            process_manager=process_manager,
            guidance="适合查看后台测试、构建或长任务的输出。若 block=true，可等待任务结束或超时。",
            prompt="用于读取通过 Bash 启动的后台任务输出。",
            read_only=True,
            destructive=False,
            supports_parallel=True,
            tags=["shell", "background", "claude_code"],
        )

    def run(self, parameters: dict):
        task_id = str(parameters.get("task_id", "")).strip()
        block = bool(parameters.get("block", False))
        timeout_ms = int(parameters.get("timeout", 0))

        if not task_id:
            return self._tool_error(
                "task_id 不能为空。",
                error_type="invalid_parameters",
                structured_data={"reason": "empty_task_id"},
            )

        try:
            snapshot = self.process_manager.get_task(
                task_id,
                block=block,
                timeout_ms=timeout_ms,
            )
        except KeyError as exc:
            return self._tool_error(
                str(exc),
                error_type="task_not_found",
                metadata={"task_id": task_id},
                structured_data={"reason": "task_not_found", "task_id": task_id},
            )

        display_text, truncated = _format_background_snapshot(snapshot, max_output_chars=self.max_output_chars)
        return self._success(snapshot, block, timeout_ms, truncated, display_text)

    def _success(self, snapshot, block: bool, timeout_ms: int, truncated: bool, display_text: str):
        return ToolResult.success(
            display_text,
            structured_data={
                **_snapshot_to_dict(snapshot),
                "truncated": truncated,
                "blocked": block,
                "timeout_ms": timeout_ms,
            },
            metadata={
                "task_id": snapshot.task_id,
                "status": snapshot.status,
                "return_code": snapshot.return_code,
                "truncated": truncated,
            },
        )


def register_task_output_tool(
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
) -> TaskOutputTool:
    tool = TaskOutputTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=process_manager,
    )
    registry.register_tool(tool)
    return tool

__all__ = [
    "TaskOutputTool",
    "register_task_output_tool",
]
