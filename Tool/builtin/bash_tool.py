"""Claude-style Bash tool backed by ProcessManager."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeBashInput
from ..runtime import BackgroundTaskSnapshot, FilesystemAccessError, FilesystemGuard, PathResolutionError, ProcessManager
from .input_normalization import normalize_generic_input


DEFAULT_BASH_OUTPUT_CHARS = 120000

BASH_TOOL_PROMPT = """用于执行本地 shell 命令，是 code agent 的高风险执行面之一。

适用场景：
- 运行测试、构建、格式化、lint、git 查询、日志排查、包管理脚本。
- 做“仓库真实状态验证”，例如确认测试是否通过、确认某个命令输出、确认生成产物。

调用前要求：
- 先明确目标是什么，再执行最小必要命令，不要把多步探索揉成一个超长命令。
- 如果命令会修改文件、安装依赖、删除内容或启动长期进程，必须清楚副作用边界。
- 需要读取某个文件后再改动时，应先读文件或用更精确的文件工具，而不是直接靠 shell 粗暴修改。

后台任务：
- 长时间运行的命令请设置 `run_in_background=true`。
- 后台启动后，使用 `TaskOutput` 轮询 stdout/stderr，再视情况用 `TaskStop` 停止。
- 不要假设后台任务已经成功完成；必须看输出或退出码。

输出解读：
- 返回里会包含 `task_id`、`status`、`return_code`、`stdout`、`stderr` 等结构化字段。
- 如果输出被截断，要根据 `truncated` 字段决定是否继续轮询或缩小命令范围。

安全边界：
- 命令会在当前工作目录执行。
- 极度危险的命令会被 guardrail 阻断。
- 这不是默认首选的文本编辑工具；精确改文件优先用 `FileEdit` / `FileWrite`。"""


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
        resource_scope: Optional[list[str]] = None,
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
            side_effect_level="high" if destructive else ("none" if read_only else "medium"),
            resource_scope=list(resource_scope or ["process", "filesystem", "workspace"]),
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
            description="执行本地 shell 命令。支持前台执行和后台任务。",
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
            resource_scope=["process", "filesystem", "workspace"],
        )

    def run(self, parameters: dict) -> ToolResult:
        command = normalize_generic_input(parameters.get("command", ""))
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
    expose_in_deferred: bool | None = True,
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
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
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
    expose_in_deferred: bool | None = True,
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
        expose_in_deferred=expose_in_deferred,
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
    registry.register_tool(task_output_tool, expose_in_deferred=expose_in_deferred)
    registry.register_tool(task_stop_tool, expose_in_deferred=expose_in_deferred)
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
