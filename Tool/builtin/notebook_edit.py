"""Claude-style NotebookEdit tool."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional
from uuid import uuid4

from ..BaseTool import ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeNotebookEditInput
from ..runtime import FilesystemAccessError, PathResolutionError, remember_file_version
from .file_write import _WorkspaceWriteTool
from .input_normalization import normalize_path_input


NOTEBOOK_EDIT_PROMPT = """用于编辑 Jupyter Notebook 的单元格内容。
- `replace` 需要定位到已有单元格。
- `insert` 若提供 `cell_id`，会在该单元格后插入；否则追加到末尾。
- `delete` 只删除目标单元格，不会自动清理其他输出或元数据。"""

MAX_NOTEBOOK_FILE_SIZE = 4 * 1024 * 1024


def _make_error(
    tool: _WorkspaceWriteTool,
    message: str,
    *,
    error_type: str,
    metadata: Optional[dict[str, Any]] = None,
    structured_data: Any = None,
) -> ToolResult:
    return tool._tool_error(message, error_type=error_type, metadata=metadata, structured_data=structured_data)


def _decode_cell_source(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _encode_cell_source(text: str, *, preserve_list: bool) -> str | list[str]:
    if preserve_list:
        return text.splitlines(keepends=True) or [text]
    return text


def _new_cell(*, cell_type: str, source: str) -> dict[str, Any]:
    cell = {
        "cell_type": cell_type,
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True) or [source],
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _normalize_cell_shape(cell: dict[str, Any], *, requested_cell_type: Optional[str] = None) -> dict[str, Any]:
    normalized = dict(cell)
    cell_type = requested_cell_type or str(normalized.get("cell_type") or "code")
    normalized["cell_type"] = cell_type
    normalized.setdefault("metadata", {})
    normalized.setdefault("id", uuid4().hex[:8])
    if cell_type == "code":
        normalized.setdefault("execution_count", None)
        normalized.setdefault("outputs", [])
    else:
        normalized.pop("execution_count", None)
        normalized.pop("outputs", None)
    return normalized


def _find_cell_index(cells: list[dict[str, Any]], cell_id: Optional[str]) -> Optional[int]:
    if not cell_id:
        return None
    for index, cell in enumerate(cells):
        if str(cell.get("id") or "") == cell_id:
            return index
    if cell_id.isdigit():
        numeric_index = int(cell_id)
        if 0 <= numeric_index < len(cells):
            return numeric_index
    return None


def _dump_notebook(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


class NotebookEditTool(_WorkspaceWriteTool):
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
    ):
        super().__init__(
            name="NotebookEdit",
            description="编辑 Jupyter Notebook 的单元格内容。",
            parameters=ClaudeNotebookEditInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            guidance="适合修改 `.ipynb` 单元格。replace/delete 需要提供可定位的 cell_id；insert 不提供 cell_id 时默认追加到末尾。",
            prompt=NOTEBOOK_EDIT_PROMPT,
            tags=["notebook", "edit", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        notebook_path = normalize_path_input(parameters.get("notebook_path", ""))
        cell_id = parameters.get("cell_id")
        new_source = str(parameters.get("new_source", ""))
        cell_type = parameters.get("cell_type")
        edit_mode = str(parameters.get("edit_mode") or "replace").strip()

        try:
            resolved = self.guard.resolve_read_path(notebook_path, cwd=self.cwd)
            if not resolved.endswith(".ipynb"):
                return _make_error(
                    self,
                    "编辑 Notebook 失败: 仅支持 `.ipynb` 文件。",
                    error_type="invalid_path",
                    metadata={"notebook_path": notebook_path},
                    structured_data={"notebook_path": notebook_path, "reason": "not_notebook"},
                )
            self.guard.ensure_file_readable(resolved)
            self.guard.ensure_file_writable(resolved)
            guard_result = self._ensure_recent_read(resolved)
            if guard_result is not None:
                return guard_result

            file_size = os.path.getsize(resolved)
            if file_size > MAX_NOTEBOOK_FILE_SIZE:
                return _make_error(
                    self,
                    f"编辑 Notebook 失败: 文件过大，当前仅允许编辑不超过 {MAX_NOTEBOOK_FILE_SIZE} 字节的 Notebook。",
                    error_type="file_too_large",
                    metadata={"notebook_path": resolved, "file_size": file_size, "max_size": MAX_NOTEBOOK_FILE_SIZE},
                    structured_data={
                        "notebook_path": resolved,
                        "reason": "file_too_large",
                        "file_size": file_size,
                        "max_size": MAX_NOTEBOOK_FILE_SIZE,
                    },
                )

            with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
                notebook = json.load(handle)
        except (PathResolutionError, FilesystemAccessError, ValueError, json.JSONDecodeError) as exc:
            return _make_error(
                self,
                f"编辑 Notebook 失败: {exc}",
                error_type="invalid_path",
                metadata={"notebook_path": notebook_path},
                structured_data={"notebook_path": notebook_path, "reason": "invalid_notebook"},
            )
        except Exception as exc:
            return _make_error(
                self,
                f"编辑 Notebook 失败: {exc}",
                error_type="notebook_edit_failed",
                metadata={"notebook_path": notebook_path},
                structured_data={"notebook_path": notebook_path, "reason": "read_failed"},
            )

        if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
            return _make_error(
                self,
                "编辑 Notebook 失败: Notebook JSON 结构无效，缺少 cells 数组。",
                error_type="invalid_notebook",
                metadata={"notebook_path": resolved},
                structured_data={"notebook_path": resolved, "reason": "invalid_notebook_structure"},
            )

        cells = notebook["cells"]
        cell_index = _find_cell_index(cells, cell_id)
        old_cell = dict(cells[cell_index]) if cell_index is not None else None

        if edit_mode in {"replace", "delete"} and cell_index is None:
            return _make_error(
                self,
                "编辑 Notebook 失败: replace/delete 模式需要提供可定位的 cell_id。",
                error_type="cell_not_found",
                metadata={"notebook_path": resolved, "cell_id": cell_id},
                structured_data={"notebook_path": resolved, "cell_id": cell_id, "reason": "cell_not_found"},
            )

        if edit_mode == "replace":
            preserve_list = isinstance(cells[cell_index].get("source"), list)
            updated_cell = _normalize_cell_shape(cells[cell_index], requested_cell_type=cell_type)
            updated_cell["source"] = _encode_cell_source(new_source, preserve_list=preserve_list)
            cells[cell_index] = updated_cell
            affected_index = cell_index
            affected_cell = updated_cell
        elif edit_mode == "insert":
            insert_cell_type = str(cell_type or "code")
            created_cell = _new_cell(cell_type=insert_cell_type, source=new_source)
            if cell_index is None:
                cells.append(created_cell)
                affected_index = len(cells) - 1
            else:
                affected_index = cell_index + 1
                cells.insert(affected_index, created_cell)
            affected_cell = created_cell
        elif edit_mode == "delete":
            affected_index = cell_index
            affected_cell = None
            cells.pop(cell_index)
        else:
            return _make_error(
                self,
                f"编辑 Notebook 失败: 不支持的 edit_mode: {edit_mode}",
                error_type="invalid_parameters",
                metadata={"notebook_path": resolved, "edit_mode": edit_mode},
                structured_data={"notebook_path": resolved, "reason": "invalid_edit_mode", "edit_mode": edit_mode},
            )

        try:
            serialized = _dump_notebook(notebook)
            self._atomic_write(resolved, serialized)
            version = remember_file_version(resolved)
        except Exception as exc:
            return _make_error(
                self,
                f"编辑 Notebook 失败: {exc}",
                error_type="notebook_edit_failed",
                metadata={"notebook_path": resolved, "edit_mode": edit_mode},
                structured_data={"notebook_path": resolved, "reason": "write_failed"},
            )

        return ToolResult.success(
            f"已更新 Notebook: {resolved}",
            structured_data={
                "notebookPath": resolved,
                "editMode": edit_mode,
                "cellId": cell_id,
                "affectedCellId": None if affected_cell is None else affected_cell.get("id"),
                "cellIndex": affected_index,
                "oldCell": old_cell,
                "newCell": affected_cell,
                "fileVersion": version.to_dict(),
            },
            metadata={
                "notebook_path": resolved,
                "edit_mode": edit_mode,
                "affected_cell_id": None if affected_cell is None else affected_cell.get("id"),
                "cell_index": affected_index,
                "file_version": version.to_dict(),
            },
        )


def register_notebook_edit_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    expose_in_deferred: bool | None = True,
) -> NotebookEditTool:
    tool = NotebookEditTool(workspace_root=workspace_root, allowed_roots=allowed_roots, cwd=cwd)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = ["NotebookEditTool", "register_notebook_edit_tool"]
