"""Local file edit tool for Claude-style coding workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..BaseTool import ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeFileEditInput
from ..runtime import FilesystemAccessError, PathResolutionError, remember_file_version
from .file_write import _WorkspaceWriteTool, _build_file_diff_payload
from .input_normalization import normalize_path_input


FILE_EDIT_PROMPT = """用于对已有文件做精确文本替换。

核心原则：
- 这是“精确编辑工具”，不是模糊编辑器。
- `old_string` 必须与目标文件中的现有文本真实匹配。
- 默认要求唯一命中；若有多处相同文本，必须显式设置 `replace_all=true`。

何时使用：
- 你已经知道要替换的原文和新文本，且只想改局部。
- 你希望尽量减少 diff，只动必要行。

何时不要用：
- 还不确定原始文本是否真实存在时，不要直接调用；先 `FileRead`。
- 需要重写整个文件结构时，优先用 `FileWrite`。
- 想做基于语义的复杂重构时，不要把一个模糊想法硬塞成 `old_string`。

失败语义：
- `no_match`：`old_string` 没命中。
- `non_unique_match`：命中多处但未设置 `replace_all=true`。
- `stale_file` / `read_required`：文件已变化或未读取过，需先重新 `FileRead`。

最佳实践：
- `old_string` 尽量包含足够上下文，避免多处重复命中。
- 先小步精确改，再运行测试或 diagnostics 验证，而不是一次拼很多大替换。"""

MAX_EDIT_FILE_SIZE = 2 * 1024 * 1024
SMART_QUOTE_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


@dataclass(frozen=True, slots=True)
class MatchPlan:
    offsets: list[int]
    match_length: int
    match_mode: str = "exact"


def _line_number_for_offset(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _preview_text(value: str, *, max_chars: int = 160) -> str:
    preview = value.replace("\n", "\\n")
    if len(preview) <= max_chars:
        return preview
    return f"{preview[:max_chars].rstrip()}..."


def _normalize_line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _canonicalize_match_text(value: str) -> str:
    return _normalize_line_endings(value).translate(SMART_QUOTE_TRANSLATION)


def _find_raw_match_offsets(content: str, needle: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index < 0:
            return offsets
        offsets.append(index)
        start = index + len(needle)


def _find_match_offsets(content: str, needle: str) -> MatchPlan:
    exact_offsets = _find_raw_match_offsets(content, needle)
    if exact_offsets:
        return MatchPlan(offsets=exact_offsets, match_length=len(needle), match_mode="exact")

    canonical_content = _canonicalize_match_text(content)
    canonical_needle = _canonicalize_match_text(needle)
    if canonical_needle != needle or canonical_content != content:
        normalized_offsets = _find_raw_match_offsets(canonical_content, canonical_needle)
        if normalized_offsets:
            return MatchPlan(
                offsets=normalized_offsets,
                match_length=len(canonical_needle),
                match_mode="normalized",
            )

    return MatchPlan(offsets=[], match_length=len(needle), match_mode="exact")


def _replace_by_offsets(content: str, offsets: list[int], match_length: int, replacement: str) -> str:
    parts: list[str] = []
    cursor = 0
    for offset in offsets:
        parts.append(content[cursor:offset])
        parts.append(replacement)
        cursor = offset + match_length
    parts.append(content[cursor:])
    return "".join(parts)


class FileEditTool(_WorkspaceWriteTool):
    """Replace one or many exact matches in an existing file."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
    ):
        super().__init__(
            name="FileEdit",
            description="对已有文件执行精确文本替换。",
            parameters=ClaudeFileEditInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="适合精确替换已有文本。默认要求 old_string 唯一命中；多处命中时请显式设置 replace_all=true。",
            prompt=FILE_EDIT_PROMPT,
            tags=["filesystem", "edit", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path", ""))
        old_string = _normalize_line_endings(str(parameters.get("old_string", "")))
        new_string = _normalize_line_endings(str(parameters.get("new_string", "")))
        replace_all = bool(parameters.get("replace_all", False))

        if not old_string:
            return self._tool_error(
                "编辑文件失败: old_string 不能为空。",
                error_type="invalid_parameters",
                metadata={"file_path": file_path},
                structured_data={"file_path": file_path, "reason": "empty_old_string"},
            )

        try:
            resolved = self.guard.resolve_read_path(file_path, cwd=self.cwd)
            self.guard.ensure_file_readable(resolved)
            self.guard.ensure_file_writable(resolved)
            guard_result = self._ensure_recent_read(resolved)
            if guard_result is not None:
                return guard_result

            file_size = os.path.getsize(resolved)
            if file_size > MAX_EDIT_FILE_SIZE:
                return self._tool_error(
                    f"编辑文件失败: 文件过大，当前仅允许编辑不超过 {MAX_EDIT_FILE_SIZE} 字节的文件。",
                    error_type="file_too_large",
                    metadata={"file_path": resolved, "file_size": file_size, "max_size": MAX_EDIT_FILE_SIZE},
                    structured_data={
                        "file_path": resolved,
                        "reason": "file_too_large",
                        "file_size": file_size,
                        "max_size": MAX_EDIT_FILE_SIZE,
                    },
                )

            with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"编辑文件失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
                structured_data={"file_path": file_path, "reason": "invalid_path"},
            )
        except Exception as exc:
            return self._tool_error(
                f"编辑文件失败: {exc}",
                error_type="file_edit_failed",
                metadata={"file_path": file_path},
                structured_data={"file_path": file_path, "reason": "read_failed"},
            )

        match_plan = _find_match_offsets(content, old_string)
        match_count = len(match_plan.offsets)
        line_numbers = [_line_number_for_offset(content, offset) for offset in match_plan.offsets[:20]]

        if match_count == 0:
            return self._tool_error(
                "编辑文件失败: old_string 未在目标文件中命中。",
                error_type="no_match",
                metadata={"file_path": resolved, "match_count": 0},
                structured_data={
                    "file_path": resolved,
                    "reason": "no_match",
                    "match_count": 0,
                    "replace_all": replace_all,
                    "old_string_preview": _preview_text(old_string),
                    "match_mode": match_plan.match_mode,
                },
            )

        if old_string == new_string:
            return ToolResult.success(
                f"文件无需更新: {resolved}",
                structured_data={
                    "file_path": resolved,
                    "changed": False,
                    "match_count": match_count,
                    "line_numbers": line_numbers,
                    "match_mode": match_plan.match_mode,
                },
                metadata={
                    "file_path": resolved,
                    "changed": False,
                    "match_count": match_count,
                    "match_mode": match_plan.match_mode,
                },
            )

        if not replace_all and match_count > 1:
            return self._tool_error(
                "编辑文件失败: old_string 命中多处，请缩小匹配范围或显式设置 replace_all=true。",
                error_type="non_unique_match",
                metadata={"file_path": resolved, "match_count": match_count},
                structured_data={
                    "file_path": resolved,
                    "reason": "multiple_matches",
                    "match_count": match_count,
                    "line_numbers": line_numbers,
                    "replace_all": replace_all,
                    "old_string_preview": _preview_text(old_string),
                    "match_mode": match_plan.match_mode,
                },
            )

        replacements = match_count if replace_all else 1
        selected_offsets = match_plan.offsets if replace_all else match_plan.offsets[:1]
        new_content = _replace_by_offsets(content, selected_offsets, match_plan.match_length, new_string)

        try:
            write_info = self._atomic_write(resolved, new_content)
            version = remember_file_version(resolved)
        except Exception as exc:
            return self._tool_error(
                f"编辑文件失败: {exc}",
                error_type="file_edit_failed",
                metadata={"file_path": resolved, "match_count": match_count},
                structured_data={
                    "file_path": resolved,
                    "reason": "write_failed",
                    "match_count": match_count,
                },
            )

        changed_lines = line_numbers if replace_all else line_numbers[:1]
        diff_payload = _build_file_diff_payload(
            resolved,
            content,
            new_content,
            workspace_root=self.workspace_root,
            created=False,
        )
        payload = {
            "file_path": resolved,
            "changed": True,
            "replacements": replacements,
            "match_count": match_count,
            "replace_all": replace_all,
            "line_numbers": changed_lines,
            "old_string_preview": _preview_text(old_string),
            "new_string_preview": _preview_text(new_string),
            "match_mode": match_plan.match_mode,
            "file_version": version.to_dict(),
            **write_info,
        }
        if diff_payload is not None:
            payload["diff"] = diff_payload
        return ToolResult.success(
            f"已更新文件: {resolved} (替换 {replacements} 处匹配)",
            structured_data=payload,
            metadata={
                "file_path": resolved,
                "replacements": replacements,
                "match_count": match_count,
                "replace_all": replace_all,
                "match_mode": match_plan.match_mode,
                "file_version": version.to_dict(),
                "diff_available": diff_payload is not None,
                **write_info,
            },
        )


def register_file_edit_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> FileEditTool:
    tool = FileEditTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = [
    "FileEditTool",
    "MAX_EDIT_FILE_SIZE",
    "register_file_edit_tool",
]
