"""Built-in code intelligence tools backed by the codeintel subsystem."""

from __future__ import annotations

import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from codeintel import CodeIntelManager, LSPCodeIntelProvider
from Tool.runtime import PathResolutionError

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import format_error_display, format_structured_display


CODEINTEL_STATUS_PROMPT = """用于检查当前工作区是否具备可用的 code intelligence / LSP 能力。
- 当你准备依赖定义跳转、引用检索、符号树或 diagnostics 之前，先用它确认可用性。
- 如果返回 `available=false`，不要继续盲目重试 codeintel 工具，直接退回 `Grep` / `FileRead` / `Glob`。
- `file_path` 最好指向当前要分析的语言文件，这样可以按语言自动挑选合适的 LSP server。"""

FIND_DEFINITION_PROMPT = """用于按源码中的“引用点”跳到定义位置。
- 只有在你已经知道具体文件和精确行列时才调用；它不是模糊搜索工具。
- `line` 和 `column` 使用 1-based；尽量把坐标放在标识符字符上，而不是空白、逗号或括号上。
- 优先用于定位函数、类、变量、方法的声明位置，再配合 `FileRead` 读取定义上下文。
- 如果结果为空，先检查坐标是否准确；若工具返回 `unavailable` 或依然找不到，再退回 `Grep` / `FileRead` / `Glob`。"""

FIND_REFERENCES_PROMPT = """用于按源码中的“定义点或引用点”查找所有引用位置。
- 适合做影响面分析、重构前摸排、确认 API 是否还在使用。
- `line` 和 `column` 使用 1-based；坐标必须落在你要跟踪的符号上。
- 默认不包含声明本身；若你希望结果里包含定义位置，显式传 `include_declaration=true`。
- 引用结果通常很多，先看结构化返回，再有选择地 `FileRead` 少量关键文件，不要把所有命中都读一遍。"""

DOCUMENT_SYMBOLS_PROMPT = """用于获取单个文件的符号树。
- 适合先快速理解一个文件的类、函数、方法、字段结构，再决定读哪些行。
- 优先在“文件很大、但你只想先看结构”时使用，而不是直接整文件 `FileRead`。
- 如果返回 `unavailable`，退回 `FileRead` 查看文件目录与关键段落。"""

WORKSPACE_SYMBOLS_PROMPT = """用于在整个工作区按符号名模糊检索。
- 适合已知道大致名称，但不知道文件位置时使用。
- 典型场景：找 `TaskService`、`SessionRestoreReport`、`FooBarManager` 这类类型或函数。
- `query` 应尽量使用较稳定的名字片段，不要把自然语言整句塞进去。
- 如果结果为空或 `unavailable`，退回 `Grep` / `Glob`。"""

DIAGNOSTICS_PROMPT = """用于读取语言服务器为某个文件生成的 diagnostics。
- 适合在修改代码前后检查报错、警告、类型问题。
- 这不是编译器替代品；它反映的是当前 LSP server 能看到的诊断结果。
- 若 diagnostics 不可用或为空，不代表代码绝对正确；必要时仍应结合测试、构建或 `FileRead` 继续分析。"""

CACHE_STATUS_PROMPT = """用于检查当前工作区 codeintel cache / offline index 的状态。
- 当你想知道当前会话是否已经做过 prewarm、缓存了多少文件、workspace symbol 离线索引是否可用时，先调用它。
- 这不是查询源码定义的工具；它返回的是缓存层状态，而不是符号结果本身。
- 如果 `offlineIndexAvailable=false`，说明当前缓存还不足以承担 workspace-level 的离线回退；这时不要假设 `GetWorkspaceSymbols` 在 provider 不可用时还能给出结果。
- 如果你准备在同一仓库里做一连串 symbol/diagnostics 查询，先看它，再决定要不要运行 `CodeIntelPrewarmWorkspace`。"""

PREWARM_WORKSPACE_PROMPT = """用于预热当前工作区的 codeintel cache / offline index。
- 它会扫描代码文件，并把 document symbols（以及可选 diagnostics）写入内存缓存，供后续 `GetWorkspaceSymbols`、`GetDocumentSymbols`、`GetDiagnostics` 在 provider 不稳定或重启后离线回退。
- 这是读操作，但代价可能不低；不要在超大仓库上无脑全量跑。优先结合 `path_prefix` 和 `max_files` 做局部预热。
- 典型场景：
  1. 你接下来要连续做很多代码查询；
  2. 你怀疑 LSP server 可能不稳定；
  3. 你想让 workspace symbol 检索具备离线快照。
- `path_prefix` 应尽量指向具体子目录，比如 `src/`, `core/`, `api/`，不要传文件路径。
- `force=false` 时，会复用仍然新鲜的缓存，避免重复扫同一批文件；只有在你明确知道代码树变化很大时，才考虑 `force=true`。"""


class CodeIntelStatusInput(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="可选。用于帮助按文件语言选择合适的 LSP server，支持相对当前工作区的路径。",
    )


class DefinitionInput(BaseModel):
    file_path: str = Field(description="要查询定义的源码文件路径，支持相对当前工作区的路径。")
    line: int = Field(ge=1, description="1-based 行号。")
    column: int = Field(ge=1, description="1-based 列号。")


class ReferencesInput(BaseModel):
    file_path: str = Field(description="要查询引用的源码文件路径，支持相对当前工作区的路径。")
    line: int = Field(ge=1, description="1-based 行号。")
    column: int = Field(ge=1, description="1-based 列号。")
    include_declaration: bool = Field(default=False, description="是否把声明/定义位置也包含进返回结果。")


class DocumentSymbolsInput(BaseModel):
    file_path: str = Field(description="要查询符号树的源码文件路径，支持相对当前工作区的路径。")


class WorkspaceSymbolsInput(BaseModel):
    query: str = Field(description="要检索的符号名或稳定片段。")
    limit: int = Field(default=50, ge=1, le=200, description="最多返回多少个符号。")


class DiagnosticsInput(BaseModel):
    file_path: str = Field(description="要拉取 diagnostics 的源码文件路径，支持相对当前工作区的路径。")


class CodeIntelCacheStatusInput(BaseModel):
    pass


class CodeIntelPrewarmWorkspaceInput(BaseModel):
    path_prefix: Optional[str] = Field(
        default=None,
        description="可选。限制预热到当前工作区下的某个子目录，比如 `src` 或 `core/runtime`。",
    )
    max_files: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="最多扫描多少个代码文件。仓库很大时应主动收紧，而不是默认全量扫描。",
    )
    include_diagnostics: bool = Field(
        default=True,
        description="是否在预热时同时拉取 diagnostics。开启后更完整，但也更慢。",
    )
    force: bool = Field(
        default=False,
        description="是否强制刷新已有缓存。默认仅在缓存缺失或文件变化时才重新扫描。",
    )


def _build_codeintel_manager(
    *,
    manager: Optional[CodeIntelManager] = None,
    provider: Optional[Any] = None,
    parent_agent: Any | None = None,
    workspace_root: Optional[str] = None,
    allowed_roots: Optional[tuple[str, ...]] = None,
) -> CodeIntelManager:
    if manager is not None:
        bind_parent = getattr(manager, "bind_parent_agent", None)
        if callable(bind_parent):
            bind_parent(parent_agent)
        return manager
    resolved_provider = provider or LSPCodeIntelProvider()
    return CodeIntelManager(
        provider=resolved_provider,
        parent_agent=parent_agent,
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
    )


def _format_query_summary(result_payload: dict[str, Any], *, action: str) -> str:
    status = str(result_payload.get("status") or "unknown")
    if status == "ok":
        count = len(list(result_payload.get("items") or []))
        return f"{action} 查询完成，返回 {count} 条结果。"
    fallback_tools = list(result_payload.get("fallbackTools") or [])
    reason = str(result_payload.get("errorMessage") or "当前 codeintel 不可用。").strip()
    if fallback_tools:
        return f"{action} 未能直接给出结果：{reason} 请改用 {fallback_tools}。"
    return f"{action} 未能直接给出结果：{reason}"


def _ephemeral_context(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "codeintel",
        "action": action,
        "status": payload.get("status"),
        "workspaceRoot": payload.get("workspaceRoot"),
        "resultCount": len(list(payload.get("items") or [])),
        "items": list(payload.get("items") or [])[:10],
        "fallbackTools": list(payload.get("fallbackTools") or []),
        "errorMessage": payload.get("errorMessage"),
    }


class _CodeIntelTool(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: type[BaseModel],
        manager: CodeIntelManager,
        guidance: str,
        prompt: str,
    ):
        self.codeintel_manager = manager
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            guidance=guidance,
            prompt=prompt,
            read_only=True,
            source="builtin",
            tags=["codeintel", "lsp", "symbol"],
            side_effect_level="none",
            resource_scope=["codeintel", "filesystem", "workspace"],
        )

    @staticmethod
    def _tool_error(message: str, *, error_type: str = "tool_error", metadata: Optional[dict[str, Any]] = None) -> ToolResult:
        return ToolResult.error(
            message,
            error_type=error_type,
            display_text=format_error_display(message, metadata),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _success_result(*, action: str, payload: dict[str, Any]) -> ToolResult:
        summary = _format_query_summary(payload, action=action)
        return ToolResult.success(
            summary,
            display_text=format_structured_display(summary, payload, payload_label="CodeIntel 返回"),
            structured_data=payload,
            ephemeral_context=_ephemeral_context(action, payload),
            metadata={
                "codeintelStatus": payload.get("status"),
                "providerName": payload.get("providerName"),
                "workspaceRoot": payload.get("workspaceRoot"),
                "resultCount": len(list(payload.get("items") or [])),
            },
        )


class CodeIntelStatusTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="CodeIntelStatus",
            description="检查当前工作区的 code intelligence / LSP 是否可用，并给出 provider、server 命令和回退建议。",
            parameters=CodeIntelStatusInput,
            manager=manager,
            guidance="先确认 codeintel 可用性，再决定是否依赖 definition/references/diagnostics；不可用时应立即退回文件级工具。",
            prompt=CODEINTEL_STATUS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = str(parameters.get("file_path") or "").strip() or None
        try:
            status = self.codeintel_manager.get_status(file_path=file_path)
            payload = status.to_dict()
            if not payload.get("available", False):
                payload["fallbackTools"] = ["FileRead", "Grep", "Glob"]
            return self._success_result(action="codeintel_status", payload=payload)
        except Exception as exc:
            return self._tool_error(
                f"检查 codeintel 状态失败: {exc}",
                error_type="codeintel_status_failed",
                metadata={"file_path": file_path},
            )


class FindDefinitionTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="FindDefinition",
            description="按文件中的精确行列定位符号定义位置，适合跳转到函数、类、变量或方法的声明处。",
            parameters=DefinitionInput,
            manager=manager,
            guidance="只在已知精确引用点时使用；若 server 不可用或返回为空，立即退回 Grep/FileRead/Glob。",
            prompt=FIND_DEFINITION_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            result = self.codeintel_manager.find_definition(
                file_path=str(parameters.get("file_path") or ""),
                line=int(parameters.get("line") or 1),
                column=int(parameters.get("column") or 1),
            )
            return self._success_result(action="find_definition", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"definition 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": parameters.get("file_path")},
            )
        except Exception as exc:
            return self._tool_error(
                f"definition 查询失败: {exc}",
                error_type="codeintel_definition_failed",
                metadata=dict(parameters),
            )


class FindReferencesTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="FindReferences",
            description="按文件中的精确行列查找符号引用位置，适合评估影响面、做重构前摸排。",
            parameters=ReferencesInput,
            manager=manager,
            guidance="结果可能很多，优先看结构化结果，再选择少量关键文件做 FileRead。",
            prompt=FIND_REFERENCES_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            result = self.codeintel_manager.find_references(
                file_path=str(parameters.get("file_path") or ""),
                line=int(parameters.get("line") or 1),
                column=int(parameters.get("column") or 1),
                include_declaration=bool(parameters.get("include_declaration")),
            )
            return self._success_result(action="find_references", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"references 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": parameters.get("file_path")},
            )
        except Exception as exc:
            return self._tool_error(
                f"references 查询失败: {exc}",
                error_type="codeintel_references_failed",
                metadata=dict(parameters),
            )


class GetDocumentSymbolsTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="GetDocumentSymbols",
            description="读取单个文件的符号树，适合先看类、函数、方法结构，再决定是否进一步读源码。",
            parameters=DocumentSymbolsInput,
            manager=manager,
            guidance="优先用于大文件结构摸排，而不是替代整文件读取。",
            prompt=DOCUMENT_SYMBOLS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            result = self.codeintel_manager.get_document_symbols(
                file_path=str(parameters.get("file_path") or ""),
            )
            return self._success_result(action="get_document_symbols", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"document symbols 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": parameters.get("file_path")},
            )
        except Exception as exc:
            return self._tool_error(
                f"document symbols 查询失败: {exc}",
                error_type="codeintel_document_symbols_failed",
                metadata=dict(parameters),
            )


class GetWorkspaceSymbolsTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="GetWorkspaceSymbols",
            description="按符号名在整个工作区检索类、函数、变量等定义，适合先定位候选文件。",
            parameters=WorkspaceSymbolsInput,
            manager=manager,
            guidance="输入尽量是稳定的名字片段，而不是自然语言整句。",
            prompt=WORKSPACE_SYMBOLS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            result = self.codeintel_manager.get_workspace_symbols(
                query=str(parameters.get("query") or ""),
                limit=int(parameters.get("limit") or 50),
            )
            return self._success_result(action="get_workspace_symbols", payload=result.to_dict())
        except Exception as exc:
            return self._tool_error(
                f"workspace symbols 查询失败: {exc}",
                error_type="codeintel_workspace_symbols_failed",
                metadata=dict(parameters),
            )


class GetDiagnosticsTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="GetDiagnostics",
            description="读取某个文件的 LSP diagnostics，适合查看报错、警告、类型问题。",
            parameters=DiagnosticsInput,
            manager=manager,
            guidance="用于查看语言服务器当前能看到的诊断结果；diagnostics 为空不代表绝对正确。",
            prompt=DIAGNOSTICS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            result = self.codeintel_manager.get_diagnostics(
                file_path=str(parameters.get("file_path") or ""),
            )
            return self._success_result(action="get_diagnostics", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"diagnostics 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": parameters.get("file_path")},
            )
        except Exception as exc:
            return self._tool_error(
                f"diagnostics 查询失败: {exc}",
                error_type="codeintel_diagnostics_failed",
                metadata=dict(parameters),
            )


class CodeIntelCacheStatusTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="CodeIntelCacheStatus",
            description="查看当前工作区 codeintel cache / offline index 的状态，包括缓存文件数、symbol 索引可用性和最近一次预热摘要。",
            parameters=CodeIntelCacheStatusInput,
            manager=manager,
            guidance="在大规模代码查询前先确认 cache 是否已预热；若离线索引还不可用，再决定是否执行 CodeIntelPrewarmWorkspace。",
            prompt=CACHE_STATUS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            payload = self.codeintel_manager.get_cache_status()
            return self._success_result(action="codeintel_cache_status", payload=payload)
        except Exception as exc:
            return self._tool_error(
                f"读取 codeintel cache 状态失败: {exc}",
                error_type="codeintel_cache_status_failed",
            )


class CodeIntelPrewarmWorkspaceTool(_CodeIntelTool):
    def __init__(self, manager: CodeIntelManager):
        super().__init__(
            name="CodeIntelPrewarmWorkspace",
            description="预热当前工作区的 codeintel cache / offline symbol index，适合在连续做很多代码查询前先建立离线快照。",
            parameters=CodeIntelPrewarmWorkspaceInput,
            manager=manager,
            guidance="优先限定 path_prefix 和 max_files，不要默认对大仓库做无边界全量预热。",
            prompt=PREWARM_WORKSPACE_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            payload = self.codeintel_manager.prewarm_workspace(
                path_prefix=str(parameters.get("path_prefix") or "").strip() or None,
                max_files=int(parameters.get("max_files") or 200),
                include_diagnostics=bool(parameters.get("include_diagnostics", True)),
                force=bool(parameters.get("force", False)),
            )
            return self._success_result(action="codeintel_prewarm_workspace", payload=payload)
        except PathResolutionError as exc:
            return self._tool_error(
                f"workspace 预热失败: {exc}",
                error_type="invalid_path",
                metadata={"path_prefix": parameters.get("path_prefix")},
            )
        except Exception as exc:
            return self._tool_error(
                f"workspace 预热失败: {exc}",
                error_type="codeintel_prewarm_failed",
                metadata=dict(parameters),
            )


def register_codeintel_tools(
    registry: ToolRegistry,
    *,
    manager: Optional[CodeIntelManager] = None,
    provider: Optional[Any] = None,
    parent_agent: Any | None = None,
    workspace_root: Optional[str] = None,
    allowed_roots: Optional[tuple[str, ...]] = None,
) -> CodeIntelManager:
    codeintel_manager = _build_codeintel_manager(
        manager=manager,
        provider=provider,
        parent_agent=parent_agent,
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
    )
    register_surface = getattr(registry, "register_runtime_surface", None)
    if callable(register_surface):
        register_surface("codeintel_manager", "default", codeintel_manager)
    registry.register_tool(CodeIntelStatusTool(codeintel_manager))
    registry.register_tool(FindDefinitionTool(codeintel_manager))
    registry.register_tool(FindReferencesTool(codeintel_manager))
    registry.register_tool(GetDocumentSymbolsTool(codeintel_manager))
    registry.register_tool(GetWorkspaceSymbolsTool(codeintel_manager))
    registry.register_tool(GetDiagnosticsTool(codeintel_manager))
    registry.register_tool(CodeIntelCacheStatusTool(codeintel_manager))
    registry.register_tool(CodeIntelPrewarmWorkspaceTool(codeintel_manager))
    return codeintel_manager


__all__ = [
    "CodeIntelCacheStatusTool",
    "CodeIntelPrewarmWorkspaceTool",
    "CodeIntelStatusTool",
    "FindDefinitionTool",
    "FindReferencesTool",
    "GetDiagnosticsTool",
    "GetDocumentSymbolsTool",
    "GetWorkspaceSymbolsTool",
    "register_codeintel_tools",
]
