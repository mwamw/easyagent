"""Local filesystem read/search tools for Claude-style coding workflows."""

from __future__ import annotations

import glob as glob_module
import fnmatch
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeFileReadInput, ClaudeGlobInput, ClaudeGrepInput, ClaudeListInput
from ..runtime import (
    FilesystemAccessError,
    FilesystemGuard,
    PathResolutionError,
    remember_file_version,
)
from .input_normalization import (
    format_no_match_message,
    glob_pattern_hints,
    grep_pattern_hints,
    normalize_generic_input,
    normalize_path_input,
    normalize_path_with_line_hint,
)


DEFAULT_MAX_RESULTS = 500
DEFAULT_MAX_HEAD_LIMIT = 250
DEFAULT_MAX_OUTPUT_CHARS = 120000

# Claude Code 2.1.88: FileReadTool/FileReadTool.ts.
FILE_READ_SECURITY_REMINDER = """<system-reminder>
Whenever you read a file, you should consider whether it would be considered malware. You CAN and SHOULD provide analysis of malware, what it is doing. But you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer questions about the code behavior.
</system-reminder>"""

FILESYSTEM_READ_PROMPT = """Reads a file from the local filesystem. You can access files within the configured workspace and allowed roots directly using this tool. If the user provides a path, use that path. It is okay to read a file that does not exist; an error will be returned.

Usage:
- file_path accepts absolute paths or paths relative to the current working directory. Do not include Markdown wrappers or a :line suffix.
- By default, it reads the whole text file, subject to the output character limit.
- You can optionally specify a 1-based line offset and limit, especially handy for long files.
- Results include line numbers starting at 1, followed by " | " and the original file content.
- This tool can extract text from PDF files (.pdf). For large PDFs (more than 20 pages), you MUST provide pages to read specific page ranges (e.g., pages: "1-5"). Maximum 20 pages per request.
- This implementation reads text and PDF text; it does not return images as visual input. Notebook files are read as their JSON source.
- This tool can only read files, not directories. To read a directory, use List.
- If you read a file that exists but has empty contents, you will receive a system reminder warning in place of file contents.
- Reading an existing file records its version for subsequent FileEdit, FileWrite, or NotebookEdit operations. If it changes externally, read it again before editing."""

GLOB_PROMPT = """- Fast file pattern matching tool.
- Supports glob patterns like "**/*.js" or "src/**/*.ts".
- Returns matching file paths sorted by modification time.
- Use this tool when you need to find files by name patterns.
- The pattern is a shell-style glob, not a regular expression. Use ** explicitly for recursive matching.
- If path is omitted, searches start at the current working directory.
- For open-ended searches that require multiple rounds of globbing and grepping, consider Agent when available."""

GREP_PROMPT = """A powerful search tool built on ripgrep, with a local fallback when ripgrep is unavailable.

Usage:
- ALWAYS use Grep for search tasks when it is available. Do not invoke grep or rg as a Bash command when this tool can accomplish the task.
- Supports regular expressions such as "log.*Error".
- Filter files with glob (e.g., "*.js", "**/*.tsx") or type (e.g., "js", "py", "rust").
- Output modes: "content" shows matching lines, "files_with_matches" shows only file paths (default), "count" shows match counts.
- Use Agent, when available, for open-ended searches requiring multiple rounds.
- Literal braces in a regex need escaping.
- By default patterns match within single lines only. Set multiline=true for cross-line patterns.
- Limit output with head_limit and offset. Use context, -A, -B, or -C when surrounding lines are needed."""

LIST_PROMPT = """Lists local directory contents.
- Use this tool to inspect a directory instead of running ls through Bash.
- If path is omitted, lists the current working directory.
- Hidden files and directories are included by default.
- Limit the path and recursion depth for large repositories.
- Set directories_only=true for a directory-only view."""


def _format_line_window(lines: list[str], start_line: int) -> str:
    width = max(4, len(str(start_line + max(len(lines) - 1, 0))))
    formatted = []
    for index, line in enumerate(lines, start=start_line):
        formatted.append(f"{index:>{width}} | {line}")
    return "\n".join(formatted)


def _clip_text(text: str, *, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    clipped = text[:max_chars].rstrip()
    return f"{clipped}\n\n...[truncated]", True


def _parse_pdf_pages(pages: Optional[str], total_pages: int) -> list[int]:
    if total_pages <= 0:
        return []
    if not pages:
        if total_pages > 20:
            raise ValueError("PDF 超过 20 页时必须显式指定 pages 范围。")
        return list(range(1, total_pages + 1))

    selected: set[int] = set()
    chunks = [chunk.strip() for chunk in pages.split(",") if chunk.strip()]
    if not chunks:
        raise ValueError("pages 参数格式无效。")
    for chunk in chunks:
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"无效页范围: {chunk}")
            for page in range(start, end + 1):
                selected.add(page)
        else:
            selected.add(int(chunk))

    normalized = sorted(selected)
    if len(normalized) > 20:
        raise ValueError("单次最多读取 20 个 PDF 页面。")
    for page in normalized:
        if page < 1 or page > total_pages:
            raise ValueError(f"PDF 页码超出范围: {page}")
    return normalized


def _normalize_workspace_root(workspace_root: Optional[str]) -> str:
    return os.path.abspath(workspace_root or os.getcwd())


def _resolve_rg_binary() -> Optional[str]:
    return shutil.which("rg")


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class _WorkspaceTool(Tool):
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
            read_only=True,
            source="builtin",
            tags=list(tags or []),
        )

    def _tool_error(self, message: str, *, error_type: str = "tool_error", metadata: Optional[dict[str, Any]] = None) -> ToolResult:
        return ToolResult.error(message, error_type=error_type, metadata=metadata)


class FileReadTool(_WorkspaceTool):
    """Read local files with line-window support."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        self.max_output_chars = max_output_chars
        super().__init__(
            name="FileRead",
            description="Read a file from the local filesystem.",
            parameters=ClaudeFileReadInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="Reads text or PDF text within the configured allowed roots.",
            prompt=FILESYSTEM_READ_PROMPT,
            tags=["filesystem", "read", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path, line_hint = normalize_path_with_line_hint(parameters.get("file_path", ""))
        offset = parameters.get("offset")
        if offset is None and line_hint is not None:
            offset = line_hint
        limit = parameters.get("limit")
        pages = parameters.get("pages")
        try:
            resolved = self.guard.resolve_read_path(file_path, cwd=self.cwd)
            suffix = Path(resolved).suffix.lower()
            if suffix == ".pdf":
                return self._read_pdf(resolved, pages)
            return self._read_text(resolved, offset=offset, limit=limit)
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"读取文件失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
            )
        except Exception as exc:
            return self._tool_error(
                f"读取文件失败: {exc}",
                error_type="file_read_failed",
                metadata={"file_path": file_path},
            )

    def _read_text(self, path: str, *, offset: Optional[int], limit: Optional[int]) -> ToolResult:
        self.guard.ensure_file_readable(path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()

        total_lines = len(lines)
        start_line = max(1, int(offset or 1))
        line_limit = int(limit) if limit is not None else total_lines
        line_limit = max(0, line_limit)
        start_index = max(0, start_line - 1)
        end_index = min(total_lines, start_index + line_limit) if line_limit else start_index
        window = lines[start_index:end_index] if line_limit else []

        display_text = _format_line_window(window, start_line) if window else ""
        display_text, truncated = _clip_text(display_text, max_chars=self.max_output_chars)
        raw_content = "\n".join(window)
        version = remember_file_version(path)

        if total_lines == 0:
            display_text = "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>"
        elif start_line > total_lines:
            display_text = (
                "<system-reminder>Warning: the file exists but is shorter than the provided "
                f"offset ({start_line}). The file has {total_lines} lines.</system-reminder>"
            )
        elif window:
            display_text += "\n\n" + FILE_READ_SECURITY_REMINDER
        if truncated:
            display_text += (
                "\n\n<system-reminder>The file output was truncated. Use FileRead with "
                "offset and limit to read the relevant remaining lines.</system-reminder>"
            )

        return ToolResult.success(
            display_text,
            structured_data={
                "file_path": path,
                "content": raw_content,
                "start_line": start_line,
                "end_line": start_line + max(len(window) - 1, 0),
                "returned_lines": len(window),
                "total_lines": total_lines,
                "truncated": truncated,
                "file_version": version.to_dict(),
            },
            metadata={
                "file_path": path,
                "total_lines": total_lines,
                "truncated": truncated,
                "file_version": version.to_dict(),
            },
        )

    def _read_pdf(self, path: str, pages: Optional[str]) -> ToolResult:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("读取 PDF 需要安装 pypdf。") from exc

        reader = PdfReader(path)
        selected_pages = _parse_pdf_pages(pages, len(reader.pages))
        chunks: list[str] = []
        for page_number in selected_pages:
            text = reader.pages[page_number - 1].extract_text() or ""
            chunks.append(f"# Page {page_number}\n{text.strip()}")

        display_text = "\n\n".join(chunks).strip()
        display_text, truncated = _clip_text(display_text, max_chars=self.max_output_chars)
        version = remember_file_version(path)

        return ToolResult.success(
            display_text,
            structured_data={
                "file_path": path,
                "pages": selected_pages,
                "content": display_text,
                "truncated": truncated,
                "file_version": version.to_dict(),
            },
            metadata={
                "file_path": path,
                "pages": selected_pages,
                "truncated": truncated,
                "file_version": version.to_dict(),
            },
        )


class ListTool(_WorkspaceTool):
    """List local directory entries with structured metadata."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        self.max_output_chars = max_output_chars
        self.max_results = max_results
        super().__init__(
            name="List",
            description="Lists local directory contents.",
            parameters=ClaudeListInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="Inspect directory structure, then use FileRead, Glob, or Grep as needed.",
            prompt=LIST_PROMPT,
            tags=["filesystem", "list", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        target = normalize_path_input(parameters.get("path")) if parameters.get("path") is not None else self.cwd
        recursive = bool(parameters.get("recursive", False))
        include_hidden = bool(parameters.get("include_hidden", True))
        directories_only = bool(parameters.get("directories_only", False))
        raw_max_depth = parameters.get("max_depth")
        max_depth = 0 if raw_max_depth is None and not recursive else raw_max_depth
        if max_depth is None and recursive:
            max_depth = 10
        max_depth = 0 if max_depth is None else max(0, int(max_depth))
        limit = min(max(int(parameters.get("limit") or self.max_results), 1), self.max_results)

        try:
            root = self.guard.resolve_directory(target, cwd=self.cwd, must_exist=True)
            entries, truncated = self._collect_entries(
                root=root,
                recursive=recursive,
                max_depth=max_depth,
                include_hidden=include_hidden,
                directories_only=directories_only,
                limit=limit,
            )
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"列出目录失败: {exc}",
                error_type="invalid_path",
                metadata={"path": target},
            )

        display_text = self._format_entries(root, entries)
        if not entries:
            display_text = f"目录为空: {root}"
        display_text, display_truncated = _clip_text(display_text, max_chars=self.max_output_chars)

        return ToolResult.success(
            display_text,
            structured_data={
                "root": root,
                "recursive": recursive,
                "max_depth": max_depth,
                "include_hidden": include_hidden,
                "directories_only": directories_only,
                "entries": entries,
                "truncated": truncated or display_truncated,
            },
            metadata={
                "root": root,
                "recursive": recursive,
                "max_depth": max_depth,
                "include_hidden": include_hidden,
                "directories_only": directories_only,
                "result_count": len(entries),
                "truncated": truncated or display_truncated,
            },
        )

    def _collect_entries(
        self,
        *,
        root: str,
        recursive: bool,
        max_depth: int,
        include_hidden: bool,
        directories_only: bool,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        entries: list[dict[str, Any]] = []
        truncated = False

        def walk(current_dir: str, depth: int) -> None:
            nonlocal truncated
            if truncated:
                return
            if depth > max_depth:
                return

            try:
                with os.scandir(current_dir) as iterator:
                    items = sorted(
                        iterator,
                        key=lambda item: (
                            not item.is_dir(follow_symlinks=False),
                            item.name.lower(),
                        ),
                    )
            except OSError as exc:
                entries.append(
                    {
                        "path": current_dir,
                        "relative_path": os.path.relpath(current_dir, root),
                        "name": os.path.basename(current_dir) or current_dir,
                        "is_dir": True,
                        "is_symlink": False,
                        "size_bytes": None,
                        "size_display": None,
                        "modified_at": None,
                        "hidden": False,
                        "depth": depth,
                        "error": str(exc),
                    }
                )
                return

            for item in items:
                if len(entries) >= limit:
                    truncated = True
                    return
                if not include_hidden and item.name.startswith("."):
                    continue

                is_dir = item.is_dir(follow_symlinks=False)
                if directories_only and not is_dir:
                    continue

                try:
                    stat_result = item.stat(follow_symlinks=False)
                    size_bytes = 0 if is_dir else int(stat_result.st_size)
                    modified_at = datetime.fromtimestamp(stat_result.st_mtime).isoformat()
                except OSError:
                    size_bytes = None
                    modified_at = None

                path = str(Path(item.path).resolve(strict=False))
                self.guard.resolver.ensure_allowed(path)
                relative_path = os.path.relpath(path, root)
                entries.append(
                    {
                        "path": path,
                        "relative_path": relative_path,
                        "name": item.name,
                        "is_dir": is_dir,
                        "is_symlink": item.is_symlink(),
                        "size_bytes": size_bytes,
                        "size_display": None if size_bytes is None else _format_size(size_bytes),
                        "modified_at": modified_at,
                        "hidden": item.name.startswith("."),
                        "depth": depth,
                    }
                )

                if recursive and is_dir:
                    walk(path, depth + 1)

        walk(root, 0)
        return entries, truncated

    @staticmethod
    def _format_entries(root: str, entries: list[dict[str, Any]]) -> str:
        lines = [f"目录: {root}"]
        for entry in entries:
            depth = int(entry.get("depth") or 0)
            indent = "  " * depth
            name = str(entry.get("name") or entry.get("relative_path") or "")
            suffix = "/" if entry.get("is_dir") else ""
            size_text = entry.get("size_display")
            line = f"{indent}{name}{suffix}"
            if size_text and not entry.get("is_dir"):
                line += f" ({size_text})"
            if entry.get("error"):
                line += f" [error: {entry['error']}]"
            lines.append(line)
        return "\n".join(lines)


class GlobTool(_WorkspaceTool):
    """Find local paths with glob patterns."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        self.max_results = max_results
        super().__init__(
            name="Glob",
            description="Finds local paths matching a glob pattern.",
            parameters=ClaudeGlobInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="Find files by name patterns before reading or searching their contents.",
            prompt=GLOB_PROMPT,
            tags=["filesystem", "search", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        original_pattern = str(parameters.get("pattern", "")).strip()
        pattern = normalize_generic_input(original_pattern)
        search_path = normalize_path_input(parameters.get("path")) if parameters.get("path") is not None else None
        if not pattern:
            return self._tool_error("pattern 不能为空。", error_type="invalid_parameters")

        try:
            root = self.guard.validate_glob_root(search_path, cwd=self.cwd)
            matches = self._match_paths(pattern, root)
            truncated = len(matches) > self.max_results
            matches = matches[: self.max_results]
            structured = [
                {
                    "path": match,
                    "relative_path": os.path.relpath(match, root),
                    "is_dir": os.path.isdir(match),
                }
                for match in matches
            ]
            hints = glob_pattern_hints(original_pattern=original_pattern, normalized_pattern=pattern)
            display_text = "\n".join(item["path"] for item in structured) or format_no_match_message(
                "未找到匹配路径。",
                scope_label="搜索根目录",
                scope_value=root,
                query_label="glob pattern",
                query_value=pattern,
                hints=hints,
            )
            return ToolResult.success(
                display_text,
                structured_data={
                    "root": root,
                    "pattern": pattern,
                    "original_pattern": original_pattern,
                    "matches": structured,
                    "truncated": truncated,
                    "hints": hints,
                },
                metadata={"root": root, "pattern": pattern, "truncated": truncated},
            )
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"Glob 失败: {exc}",
                error_type="invalid_path",
                metadata={"pattern": pattern, "path": search_path},
            )

    def _match_paths(self, pattern: str, root: str) -> list[str]:
        if os.path.isabs(pattern):
            self.guard.resolver.ensure_allowed(pattern)
            raw_matches = glob_module.glob(pattern, recursive=True)
        else:
            raw_matches = glob_module.glob(os.path.join(root, pattern), recursive=True)
        normalized = sorted(str(Path(item).resolve(strict=False)) for item in raw_matches)
        unique_matches: list[str] = []
        seen: set[str] = set()
        for match in normalized:
            if match in seen:
                continue
            self.guard.resolver.ensure_allowed(match)
            seen.add(match)
            unique_matches.append(match)
        return unique_matches


class GrepTool(_WorkspaceTool):
    """Search local file contents using rg or a Python fallback."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        rg_binary: Optional[str] = None,
        max_head_limit: int = DEFAULT_MAX_HEAD_LIMIT,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        self.rg_binary = rg_binary or _resolve_rg_binary()
        self.max_head_limit = max_head_limit
        self.max_output_chars = max_output_chars
        super().__init__(
            name="Grep",
            description="Searches local file contents for a pattern.",
            parameters=ClaudeGrepInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="Find symbols, settings, and call sites. Returns matching file paths by default.",
            prompt=GREP_PROMPT,
            tags=["filesystem", "search", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        original_pattern = str(parameters.get("pattern", "")).strip()
        pattern = normalize_generic_input(original_pattern)
        path_value = normalize_path_input(parameters.get("path")) if parameters.get("path") is not None else None
        if not pattern:
            return self._tool_error("pattern 不能为空。", error_type="invalid_parameters")

        try:
            normalized_parameters = dict(parameters)
            normalized_parameters["glob"] = normalize_generic_input(parameters.get("glob")) if parameters.get("glob") else None
            search_path = self._resolve_search_path(path_value)
            if self.rg_binary:
                return self._run_rg(pattern, search_path, normalized_parameters, original_pattern=original_pattern)
            return self._run_python_fallback(pattern, search_path, normalized_parameters, original_pattern=original_pattern)
        except (PathResolutionError, FilesystemAccessError, ValueError, re.error) as exc:
            return self._tool_error(
                f"Grep 失败: {exc}",
                error_type="grep_failed",
                metadata={"pattern": pattern, "path": path_value},
            )

    def _resolve_search_path(self, path_value: Optional[str]) -> str:
        candidate = path_value or self.cwd
        resolved = self.guard.resolver.resolve(candidate, cwd=self.cwd, must_exist=True)
        if not (os.path.isdir(resolved) or os.path.isfile(resolved)):
            raise ValueError(f"搜索路径不是文件或目录: {resolved}")
        return resolved

    def _normalized_head_limit(self, raw_value: Optional[int]) -> Optional[int]:
        if raw_value is None:
            return self.max_head_limit
        value = int(raw_value)
        if value == 0:
            return None
        return max(0, min(value, self.max_head_limit))

    def _apply_offset_and_limit(self, items: list[str], *, offset: int, head_limit: Optional[int]) -> list[str]:
        sliced = items[max(0, offset):]
        if head_limit is None:
            return sliced
        return sliced[:head_limit]

    def _run_rg(self, pattern: str, search_path: str, parameters: dict, *, original_pattern: str) -> ToolResult:
        output_mode = parameters.get("output_mode") or "files_with_matches"
        offset = int(parameters.get("offset") or 0)
        head_limit = self._normalized_head_limit(parameters.get("head_limit"))
        command = [self.rg_binary, "--color", "never"]

        if output_mode == "files_with_matches":
            command.append("-l")
        elif output_mode == "count":
            command.append("-c")
        else:
            command.append("--no-heading")
            if parameters.get("line_numbers", True):
                command.append("-n")
            before_context = parameters.get("before_context")
            after_context = parameters.get("after_context")
            full_context = parameters.get("full_context") or parameters.get("context")
            if before_context is not None:
                command.extend(["-B", str(before_context)])
            if after_context is not None:
                command.extend(["-A", str(after_context)])
            if full_context is not None:
                command.extend(["-C", str(full_context)])

        if parameters.get("ignore_case"):
            command.append("-i")
        if parameters.get("glob"):
            command.extend(["--glob", str(parameters["glob"])])
        if parameters.get("type"):
            command.extend(["--type", str(parameters["type"])])
        if parameters.get("multiline"):
            command.extend(["-U", "--multiline-dotall"])

        command.extend([pattern, search_path])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)

        if completed.returncode not in {0, 1}:
            raise RuntimeError(completed.stderr.strip() or "rg 执行失败。")

        raw_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        normalized_lines, match_count = self._normalize_rg_output(raw_lines, output_mode)
        visible_lines = self._apply_offset_and_limit(normalized_lines, offset=offset, head_limit=head_limit)
        hints = grep_pattern_hints(
            original_pattern=original_pattern,
            normalized_pattern=pattern,
            file_glob=parameters.get("glob"),
        )
        display_text = "\n".join(visible_lines) or format_no_match_message(
            "未找到匹配内容。",
            scope_label="搜索路径",
            scope_value=search_path,
            query_label="grep pattern",
            query_value=pattern,
            hints=hints,
        )
        display_text, truncated = _clip_text(display_text, max_chars=self.max_output_chars)
        return ToolResult.success(
            display_text,
            structured_data={
                "pattern": pattern,
                "original_pattern": original_pattern,
                "path": search_path,
                "output_mode": output_mode,
                "matches": visible_lines,
                "truncated": truncated or len(visible_lines) != len(normalized_lines),
                "match_count": match_count,
                "result_count": len(normalized_lines),
                "engine": "rg",
                "hints": hints,
            },
            metadata={
                "pattern": pattern,
                "path": search_path,
                "output_mode": output_mode,
                "match_count": match_count,
                "result_count": len(normalized_lines),
                "engine": "rg",
            },
        )

    def _normalize_rg_output(self, raw_lines: list[str], output_mode: str) -> tuple[list[str], int]:
        if output_mode == "content":
            return raw_lines, len(raw_lines)

        normalized_lines: list[str] = []
        match_count = 0

        for line in raw_lines:
            if output_mode == "count":
                path_part, separator, count_part = line.rpartition(":")
                if separator and count_part.isdigit():
                    normalized_path = path_part if os.path.isabs(path_part) else os.path.abspath(path_part)
                    normalized_lines.append(f"{normalized_path}:{count_part}")
                    match_count += int(count_part)
                    continue
                if line.isdigit():
                    normalized_lines.append(line)
                    match_count += int(line)
                    continue

            normalized_line = line if os.path.isabs(line) else os.path.abspath(line)
            normalized_lines.append(normalized_line)
            match_count += 1

        return normalized_lines, match_count

    def _run_python_fallback(self, pattern: str, search_path: str, parameters: dict, *, original_pattern: str) -> ToolResult:
        output_mode = parameters.get("output_mode") or "files_with_matches"
        regex_flags = re.MULTILINE
        if parameters.get("ignore_case"):
            regex_flags |= re.IGNORECASE
        if parameters.get("multiline"):
            regex_flags |= re.DOTALL
        compiled = re.compile(pattern, regex_flags)
        if os.path.isfile(search_path):
            files = [search_path] if self._matches_glob(search_path, search_path, parameters.get("glob")) else []
        else:
            files = self._iter_files(search_path, parameters.get("glob"))
        raw_results: list[str] = []
        match_count = 0

        for file_path in files:
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()

            if output_mode == "files_with_matches":
                if compiled.search(content):
                    raw_results.append(file_path)
                    match_count += 1
            elif output_mode == "count":
                count = len(compiled.findall(content))
                if count:
                    raw_results.append(f"{file_path}:{count}")
                    match_count += count
            else:
                lines = content.splitlines()
                for line_number, line in enumerate(lines, start=1):
                    if compiled.search(line):
                        if parameters.get("line_numbers", True):
                            raw_results.append(f"{file_path}:{line_number}:{line}")
                        else:
                            raw_results.append(f"{file_path}:{line}")
                        match_count += 1

        offset = int(parameters.get("offset") or 0)
        head_limit = self._normalized_head_limit(parameters.get("head_limit"))
        visible = self._apply_offset_and_limit(raw_results, offset=offset, head_limit=head_limit)
        hints = grep_pattern_hints(
            original_pattern=original_pattern,
            normalized_pattern=pattern,
            file_glob=parameters.get("glob"),
        )
        display_text = "\n".join(visible) or format_no_match_message(
            "未找到匹配内容。",
            scope_label="搜索路径",
            scope_value=search_path,
            query_label="grep pattern",
            query_value=pattern,
            hints=hints,
        )
        display_text, truncated = _clip_text(display_text, max_chars=self.max_output_chars)
        return ToolResult.success(
            display_text,
            structured_data={
                "pattern": pattern,
                "original_pattern": original_pattern,
                "path": search_path,
                "output_mode": output_mode,
                "matches": visible,
                "truncated": truncated or len(visible) != len(raw_results),
                "match_count": match_count,
                "result_count": len(raw_results),
                "engine": "python",
                "hints": hints,
            },
            metadata={
                "pattern": pattern,
                "path": search_path,
                "output_mode": output_mode,
                "match_count": match_count,
                "result_count": len(raw_results),
                "engine": "python",
            },
        )

    def _iter_files(self, root: str, glob_pattern: Optional[str]) -> list[str]:
        matches: list[str] = []
        for current_root, _, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(current_root, filename)
                if not self._matches_glob(path, root, glob_pattern):
                    continue
                matches.append(path)
        return sorted(matches)

    @staticmethod
    def _matches_glob(path: str, root: str, glob_pattern: Optional[str]) -> bool:
        if not glob_pattern:
            return True
        relative_path = os.path.relpath(path, root)
        return fnmatch.fnmatch(relative_path, glob_pattern) or fnmatch.fnmatch(os.path.basename(path), glob_pattern)


def register_file_read_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> FileReadTool:
    tool = FileReadTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_list_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> ListTool:
    tool = ListTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_glob_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> GlobTool:
    tool = GlobTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_grep_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    rg_binary: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> GrepTool:
    tool = GrepTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        rg_binary=rg_binary,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_filesystem_tools(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    rg_binary: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> tuple[FileReadTool, ListTool, GlobTool, GrepTool]:
    file_read = register_file_read_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd, expose_in_deferred=expose_in_deferred)
    list_tool = register_list_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd, expose_in_deferred=expose_in_deferred)
    glob_tool = register_glob_tool(registry, workspace_root, allowed_roots=allowed_roots, cwd=cwd, expose_in_deferred=expose_in_deferred)
    grep_tool = register_grep_tool(
        registry,
        workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        rg_binary=rg_binary,
        expose_in_deferred=expose_in_deferred,
    )
    return file_read, list_tool, glob_tool, grep_tool


__all__ = [
    "FileReadTool",
    "ListTool",
    "GlobTool",
    "GrepTool",
    "register_file_read_tool",
    "register_list_tool",
    "register_glob_tool",
    "register_grep_tool",
    "register_filesystem_tools",
]
