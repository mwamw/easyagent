"""Local file write tool for Claude-style coding workflows."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeFileWriteInput
from ..runtime import (
    FilesystemAccessError,
    FilesystemGuard,
    PathResolutionError,
    recorded_file_is_current,
    remember_file_version,
)


FILE_WRITE_PROMPT = """用于创建新文件或整体覆盖已有文件。
- 当你已经确定完整文件内容时使用。
- 会直接替换目标文件当前内容；若只想做局部替换，优先使用 `FileEdit`。
- 路径必须位于允许的工作区内。"""


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


def write_text_atomically(path: str, content: str, *, workspace_root: str) -> dict[str, Any]:
    """Write text to a file atomically while preserving mode when possible."""
    parent = os.path.dirname(path) or workspace_root
    existing_mode: Optional[int] = None
    created = not os.path.exists(path)

    if not created:
        existing_mode = os.stat(path).st_mode

    fd, temp_path = tempfile.mkstemp(prefix=".easyagent-", suffix=".tmp", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)

        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return {
        "created": created,
        "chars_written": len(content),
        "bytes_written": len(content.encode("utf-8")),
    }


class _WorkspaceWriteTool(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: type[BaseModel],
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        guidance: str = "",
        prompt: str = "",
        tags: Optional[list[str]] = None,
    ):
        self.workspace_root = _normalize_workspace_root(workspace_root)
        self.cwd = os.path.abspath(cwd or self.workspace_root)
        self.guard = FilesystemGuard(self.workspace_root, allowed_roots=allowed_roots)
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            guidance=guidance,
            prompt=prompt,
            read_only=False,
            destructive=True,
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

    def _resolve_write_target(self, file_path: str) -> str:
        resolved = self.guard.resolve_write_path(file_path, cwd=self.cwd, create_parents=False)
        if os.path.isdir(resolved):
            raise ValueError(f"目标路径不能是目录: {resolved}")
        if os.path.exists(resolved):
            self.guard.ensure_file_writable(resolved)
        else:
            self.guard.ensure_parent_writable(resolved)
        return resolved

    def _atomic_write(self, path: str, content: str) -> dict[str, Any]:
        return write_text_atomically(path, content, workspace_root=self.workspace_root)

    def _ensure_recent_read(self, path: str) -> ToolResult | None:
        if not os.path.exists(path):
            return None

        is_current, recorded_version, current_version = recorded_file_is_current(path)
        if is_current:
            return None

        if recorded_version is None:
            message = f"修改已有文件前请先使用 FileRead 读取当前文件: {path}"
            error_type = "read_required"
            reason = "file_not_read"
        else:
            message = f"文件在上次读取后已发生变化，请先重新使用 FileRead 读取最新内容: {path}"
            error_type = "stale_file"
            reason = "stale_read"

        return self._tool_error(
            message,
            error_type=error_type,
            metadata={
                "file_path": path,
                "reason": reason,
                "current_version": current_version.to_dict(),
                "recorded_version": recorded_version.to_dict() if recorded_version else None,
            },
            structured_data={
                "file_path": path,
                "reason": reason,
                "current_version": current_version.to_dict(),
                "recorded_version": recorded_version.to_dict() if recorded_version else None,
            },
        )


class FileWriteTool(_WorkspaceWriteTool):
    """Create or overwrite a whole file."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
    ):
        super().__init__(
            name="FileWrite",
            description="创建新文件或整体覆盖已有文件内容。",
            parameters=ClaudeFileWriteInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="适合一次性生成完整文件；如果只想替换局部内容，优先使用 FileEdit。",
            prompt=FILE_WRITE_PROMPT,
            tags=["filesystem", "write", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = str(parameters.get("file_path", "")).strip()
        content = str(parameters.get("content", ""))

        try:
            resolved = self._resolve_write_target(file_path)
            guard_result = self._ensure_recent_read(resolved)
            if guard_result is not None:
                return guard_result
            write_info = self._atomic_write(resolved, content)
            version = remember_file_version(resolved)
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"写入文件失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
                structured_data={"file_path": file_path, "reason": "invalid_path"},
            )
        except Exception as exc:
            return self._tool_error(
                f"写入文件失败: {exc}",
                error_type="file_write_failed",
                metadata={"file_path": file_path},
                structured_data={"file_path": file_path, "reason": "write_failed"},
            )

        created = write_info["created"]
        action = "已创建文件" if created else "已覆盖文件"
        return ToolResult.success(
            f"{action}: {resolved}",
            structured_data={
                "file_path": resolved,
                **write_info,
                "file_version": version.to_dict(),
            },
            metadata={"file_path": resolved, **write_info, "file_version": version.to_dict()},
        )


def register_file_write_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
) -> FileWriteTool:
    tool = FileWriteTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool)
    return tool


__all__ = [
    "FileWriteTool",
    "register_file_write_tool",
    "write_text_atomically",
]
