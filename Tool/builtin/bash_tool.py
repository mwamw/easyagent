"""Claude-style Bash tool backed by ProcessManager."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeBashInput
from ..runtime import BackgroundTaskSnapshot, FilesystemAccessError, FilesystemGuard, PathResolutionError, ProcessManager


DEFAULT_BASH_OUTPUT_CHARS = 120000

BASH_TOOL_PROMPT = """用于执行本地 shell 命令。
- 优先用于测试、构建、格式化、搜索日志、git 查询等本地开发任务。
- 长时间运行的命令请设置 `run_in_background=true`，然后用 `TaskOutput` 轮询输出。
- 命令会在当前工作目录执行；如果命令有副作用，请确保目标文件已经读取并理解。"""


def _normalize_workspace_root(workspace_root: Optional[str]) -> str:
    return os.path.abspath(workspace_root or os.getcwd())


def _structured_error(
    message: str,
    *,
    error_type: str,
    metadata: Optional[dict[str, Any]] = None,
    structured_data: Any = None,
) -> ToolResult:
    return ToolResult(
        status="error",
        content=message,
        error_type=error_type,
        metadata=dict(metadata or {}),
        structured_data=structured_data,
    )


def _clip_text(value: str, *, max_chars: int = DEFAULT_BASH_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    clipped = value[:max_chars].rstrip()
    return f"{clipped}\n\n...[truncated]", True


def _stringify_command(command: Any) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def _snapshot_to_dict(snapshot: BackgroundTaskSnapshot) -> dict[str, Any]:
    return {
        "task_id": snapshot.task_id,
        "command": snapshot.command,
        "cwd": snapshot.cwd,
        "status": snapshot.status,
        "return_code": snapshot.return_code,
        "stdout": snapshot.stdout,
        "stderr": snapshot.stderr,
        "started_at": snapshot.started_at,
        "finished_at": snapshot.finished_at,
    }


def _format_background_snapshot(snapshot: BackgroundTaskSnapshot, *, max_output_chars: int) -> tuple[str, bool]:
    sections = [
        f"任务 ID: {snapshot.task_id}",
        f"状态: {snapshot.status}",
        f"命令: {_stringify_command(snapshot.command)}",
        f"cwd: {snapshot.cwd}",
    ]
    if snapshot.return_code is not None:
        sections.append(f"退出码: {snapshot.return_code}")
    if snapshot.stdout:
        sections.append(f"stdout:\n{snapshot.stdout}")
    if snapshot.stderr:
        sections.append(f"stderr:\n{snapshot.stderr}")
    display_text = "\n\n".join(sections)
    return _clip_text(display_text, max_chars=max_output_chars)


class _ShellToolBase(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: type[BaseModel],
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        shell: str = "bash",
        command_timeout_ms: int = 120000,
        max_background_tasks: int = 8,
        max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
        process_manager: Optional[ProcessManager] = None,
        guidance: str = "",
        prompt: str = "",
        read_only: bool = False,
        destructive: bool = True,
        supports_parallel: bool = True,
        tags: Optional[list[str]] = None,
    ):
        self.workspace_root = _normalize_workspace_root(workspace_root)
        self.guard = FilesystemGuard(self.workspace_root, allowed_roots=allowed_roots)
        self.cwd = self.guard.resolve_directory(cwd or self.workspace_root, cwd=self.workspace_root, must_exist=True)
        self.max_output_chars = max_output_chars
        self.command_timeout_ms = command_timeout_ms
        self.process_manager = process_manager or ProcessManager(shell=shell, max_background_tasks=max_background_tasks)
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            guidance=guidance,
            prompt=prompt,
            read_only=read_only,
            destructive=destructive,
            supports_parallel=supports_parallel,
            source="builtin",
            tags=list(tags or []),
        )

    def _tool_error(
        self,
        message: str,
        *,
        error_type: str = "tool_error",
        metadata: Optional[dict[str, Any]] = None,
        structured_data: Any = None,
    ) -> ToolResult:
        return _structured_error(
            message,
            error_type=error_type,
            metadata=metadata,
            structured_data=structured_data,
        )


class BashTool(_ShellToolBase):
    """Execute shell commands in the local workspace."""

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
            name="Bash",
            description="执行本地 shell 命令，支持前台执行和后台任务。",
            parameters=ClaudeBashInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            shell=shell,
            command_timeout_ms=command_timeout_ms,
            max_background_tasks=max_background_tasks,
            max_output_chars=max_output_chars,
            process_manager=process_manager,
            guidance="适合运行测试、构建、格式化、脚本和 git 查询。长任务请放后台，然后使用 TaskOutput 查看输出。",
            prompt=BASH_TOOL_PROMPT,
            read_only=False,
            destructive=True,
            supports_parallel=False,
            tags=["shell", "local", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        command = str(parameters.get("command", "")).strip()
        description = parameters.get("description")
        run_in_background = bool(parameters.get("run_in_background", False))
        dangerously_disable_sandbox = bool(parameters.get("dangerouslyDisableSandbox", False))
        timeout_ms = parameters.get("timeout")
        timeout_ms = self.command_timeout_ms if timeout_ms is None else int(timeout_ms)

        if not command:
            return self._tool_error(
                "命令不能为空。",
                error_type="invalid_parameters",
                structured_data={"reason": "empty_command"},
            )

        if dangerously_disable_sandbox:
            return self._tool_error(
                "当前 EasyAgent Bash 工具还不支持在工具层切换 dangerouslyDisableSandbox；请由宿主运行时处理权限模式。",
                error_type="unsupported_option",
                metadata={"command": command, "dangerouslyDisableSandbox": True},
                structured_data={"reason": "unsupported_option", "option": "dangerouslyDisableSandbox"},
            )

        try:
            if run_in_background:
                snapshot = self.process_manager.start_background(
                    command,
                    cwd=self.cwd,
                    use_shell=True,
                )
                display_text, truncated = _format_background_snapshot(snapshot, max_output_chars=self.max_output_chars)
                return ToolResult.success(
                    display_text,
                    structured_data={
                        **_snapshot_to_dict(snapshot),
                        "description": description,
                        "truncated": truncated,
                    },
                    metadata={
                        "task_id": snapshot.task_id,
                        "status": snapshot.status,
                        "cwd": snapshot.cwd,
                        "description": description,
                        "truncated": truncated,
                    },
                )

            result = self.process_manager.run(
                command,
                cwd=self.cwd,
                timeout_ms=timeout_ms,
                use_shell=True,
            )
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"命令执行失败: {exc}",
                error_type="invalid_path",
                metadata={"command": command, "cwd": self.cwd},
                structured_data={"reason": "invalid_path", "command": command, "cwd": self.cwd},
            )
        except Exception as exc:
            return self._tool_error(
                f"命令执行失败: {exc}",
                error_type="command_failed",
                metadata={"command": command, "cwd": self.cwd},
                structured_data={"reason": "command_failed", "command": command, "cwd": self.cwd},
            )

        sections = [
            f"命令: {command}",
            f"cwd: {result.cwd}",
            f"退出码: {result.return_code}",
        ]
        if result.timed_out:
            sections.append("状态: timed_out")
        if result.stdout:
            sections.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            sections.append(f"stderr:\n{result.stderr}")
        display_text, truncated = _clip_text("\n\n".join(sections), max_chars=self.max_output_chars)
        return ToolResult.success(
            display_text,
            structured_data={
                "command": command,
                "cwd": result.cwd,
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "description": description,
                "truncated": truncated,
            },
            metadata={
                "command": command,
                "cwd": result.cwd,
                "return_code": result.return_code,
                "timed_out": result.timed_out,
                "description": description,
                "truncated": truncated,
            },
        )


def register_bash_tool(
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
) -> BashTool:
    tool = BashTool(
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


def register_shell_tools(
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
):
    from .task_output import TaskOutputTool
    from .task_stop import TaskStopTool

    manager = process_manager or ProcessManager(shell=shell, max_background_tasks=max_background_tasks)
    bash_tool = register_bash_tool(
        registry,
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=manager,
    )
    task_output_tool = TaskOutputTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=manager,
    )
    task_stop_tool = TaskStopTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=manager,
    )
    registry.register_tool(task_output_tool)
    registry.register_tool(task_stop_tool)
    return bash_tool, task_output_tool, task_stop_tool


__all__ = [
    "BashTool",
    "DEFAULT_BASH_OUTPUT_CHARS",
    "_ShellToolBase",
    "_clip_text",
    "_snapshot_to_dict",
    "_format_background_snapshot",
    "register_bash_tool",
    "register_shell_tools",
]
