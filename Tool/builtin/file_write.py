"""Local file write tool for Claude-style coding workflows."""

from __future__ import annotations

import difflib
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
from .input_normalization import normalize_path_input


FILE_WRITE_PROMPT = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use FileRead first to read its contents. This tool will fail if you did not read it first or it changed since that read.
- Prefer FileEdit for modifying existing files — it only sends the replacement. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the user.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.
- Paths must be within the configured workspace or allowed roots. Writes use atomic replacement."""


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


def _normalize_diff_path(path: str) -> str:
    return path.replace(os.sep, "/")


def _diff_relative_path(path: str, workspace_root: str) -> str:
    try:
        relative = os.path.relpath(path, workspace_root)
    except ValueError:
        return _normalize_diff_path(path)
    if relative in ("", "."):
        return _normalize_diff_path(os.path.basename(path) or path)
    if relative == os.pardir or relative.startswith(f"..{os.sep}"):
        return _normalize_diff_path(path)
    return _normalize_diff_path(relative)


def _build_file_diff_payload(
    path: str,
    before: str,
    after: str,
    *,
    workspace_root: str,
    created: bool,
) -> Optional[dict[str, Any]]:
    if before == after:
        return None
    relative_path = _diff_relative_path(path, workspace_root)
    before_label = "/dev/null" if created else f"a/{relative_path}"
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=before_label,
            tofile=f"b/{relative_path}",
            lineterm="",
        )
    )
    lines = [f"diff --git a/{relative_path} b/{relative_path}"]
    if created:
        lines.append("new file mode 100644")
    lines.extend(diff_lines)
    return {
        "file_path": path,
        "relative_path": relative_path,
        "created": created,
        "unified": "\n".join(lines),
    }


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
            side_effect_level="high",
            resource_scope=["filesystem", "workspace"],
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
            description="Writes a file to the local filesystem.",
            parameters=ClaudeFileWriteInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="Use for new files or complete rewrites; prefer FileEdit for local changes.",
            prompt=FILE_WRITE_PROMPT,
            tags=["filesystem", "write", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path", ""))
        content = str(parameters.get("content", ""))

        try:
            resolved = self._resolve_write_target(file_path)
            guard_result = self._ensure_recent_read(resolved)
            if guard_result is not None:
                return guard_result
            before_content = ""
            if os.path.exists(resolved):
                with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
                    before_content = handle.read()
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
        diff_payload = _build_file_diff_payload(
            resolved,
            before_content,
            content,
            workspace_root=self.workspace_root,
            created=created,
        )
        payload = {
            "file_path": resolved,
            **write_info,
            "file_version": version.to_dict(),
        }
        if diff_payload is not None:
            payload["diff"] = diff_payload
        return ToolResult.success(
            f"{action}: {resolved}",
            structured_data=payload,
            metadata={
                "file_path": resolved,
                **write_info,
                "file_version": version.to_dict(),
                "diff_available": diff_payload is not None,
            },
        )


def register_file_write_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> FileWriteTool:
    tool = FileWriteTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = [
    "FileWriteTool",
    "register_file_write_tool",
    "write_text_atomically",
]
