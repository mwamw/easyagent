"""Built-in code intelligence tools backed by the codeintel subsystem."""

from __future__ import annotations

import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from codeintel import CodeIntelManager, LSPCodeIntelProvider
from Tool.runtime import PathResolutionError

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .input_normalization import normalize_generic_input, normalize_path_input
from .display_utils import format_error_display, format_structured_display


CODEINTEL_STATUS_PROMPT = """Checks whether code intelligence / LSP is available in the current workspace.
- Check availability before relying on definitions, references, symbols, or diagnostics.
- If available=false, use Grep, FileRead, or Glob instead of repeatedly retrying.
- Supply a relevant source file_path to select the server for its language."""

FIND_DEFINITION_PROMPT = """Finds a symbol's definition from a precise source position.
- Use only when the file and exact line and column are known.
- line and column are 1-based and should point to the identifier.
- Read the returned definition with FileRead for context.
- If no result is found, check the coordinates; if unavailable, use Grep, FileRead, or Glob."""

FIND_REFERENCES_PROMPT = """Finds references to a symbol at a source position.
- Use for impact analysis and before refactoring.
- line and column are 1-based and must point to the target symbol.
- Set include_declaration=true to include its declaration.
- Inspect the structured results and read selected relevant files, not every match."""

DOCUMENT_SYMBOLS_PROMPT = """Returns the symbol tree for one source file.
- Use to understand its classes, functions, methods, and fields before reading selected sections.
- Useful for exploring the structure of a large file.
- If unavailable, inspect the file with FileRead."""

WORKSPACE_SYMBOLS_PROMPT = """Searches workspace symbols by name.
- Use when the approximate symbol name is known but its file is not.
- Supply a stable name fragment, not an entire natural-language question.
- If unavailable or empty, fall back to Grep or Glob."""

DIAGNOSTICS_PROMPT = """Reads the language server's diagnostics for a source file.
- Use before or after changes to inspect errors, warnings, and type issues.
- This is not a replacement for a compiler or test suite.
- Empty or unavailable diagnostics do not prove the code is correct."""

CACHE_STATUS_PROMPT = """Reports the code intelligence cache and offline index status.
- Inspect cached file counts, offline symbol index availability, and the last prewarm summary.
- This returns cache status, not source definitions.
- If offlineIndexAvailable=false, do not assume workspace symbol queries can work without a provider.
- Use before deciding whether CodeIntelPrewarmWorkspace is needed."""

PREWARM_WORKSPACE_PROMPT = """Prewarms the workspace code intelligence cache and offline symbol index.
- Scans source files and caches document symbols and, optionally, diagnostics.
- Bound the work with path_prefix and max_files, especially in a large repository.
- path_prefix should identify a directory, not a file.
- force=false reuses fresh cached entries. Use force=true only when a refresh is needed.
- This reads source files but may be expensive; use it when subsequent queries benefit from the cache."""


class CodeIntelStatusInput(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="Optional file path used to select a language server; may be workspace-relative.",
    )


class DefinitionInput(BaseModel):
    file_path: str = Field(description="Source file containing the reference; may be workspace-relative.")
    line: int = Field(ge=1, description="Line number (1-based).")
    column: int = Field(ge=1, description="Column number (1-based).")


class ReferencesInput(BaseModel):
    file_path: str = Field(description="Source file containing the symbol; may be workspace-relative.")
    line: int = Field(ge=1, description="Line number (1-based).")
    column: int = Field(ge=1, description="Column number (1-based).")
    include_declaration: bool = Field(default=False, description="Include the declaration or definition in results.")


class DocumentSymbolsInput(BaseModel):
    file_path: str = Field(description="Source file whose symbol tree should be inspected; may be workspace-relative.")


class WorkspaceSymbolsInput(BaseModel):
    query: str = Field(description="Symbol name or stable name fragment.")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of symbols.")


class DiagnosticsInput(BaseModel):
    file_path: str = Field(description="Source file whose diagnostics should be retrieved; may be workspace-relative.")


class CodeIntelCacheStatusInput(BaseModel):
    pass


class CodeIntelPrewarmWorkspaceInput(BaseModel):
    path_prefix: Optional[str] = Field(
        default=None,
        description="Optional workspace subdirectory to prewarm, such as src or core/runtime.",
    )
    max_files: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum source files to scan. Use a smaller bound in large repositories.",
    )
    include_diagnostics: bool = Field(
        default=True,
        description="Include diagnostics during prewarming; more complete but slower.",
    )
    force: bool = Field(
        default=False,
        description="Force cache refresh; normally only missing or changed files are scanned.",
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
            description="Checks workspace code intelligence availability, server configuration, and fallback options.",
            parameters=CodeIntelStatusInput,
            manager=manager,
            guidance="Check availability before relying on LSP queries; use file tools if unavailable.",
            prompt=CODEINTEL_STATUS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path") or "") or None
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
            description="Finds a symbol definition at an exact source line and column.",
            parameters=DefinitionInput,
            manager=manager,
            guidance="Use known coordinates; fall back to Grep, FileRead, or Glob if unavailable.",
            prompt=FIND_DEFINITION_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path") or "")
        try:
            result = self.codeintel_manager.find_definition(
                file_path=file_path,
                line=int(parameters.get("line") or 1),
                column=int(parameters.get("column") or 1),
            )
            return self._success_result(action="find_definition", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"definition 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
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
            description="Finds references to a symbol at an exact source line and column.",
            parameters=ReferencesInput,
            manager=manager,
            guidance="Inspect structured results before reading selected relevant files.",
            prompt=FIND_REFERENCES_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path") or "")
        try:
            result = self.codeintel_manager.find_references(
                file_path=file_path,
                line=int(parameters.get("line") or 1),
                column=int(parameters.get("column") or 1),
                include_declaration=bool(parameters.get("include_declaration")),
            )
            return self._success_result(action="find_references", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"references 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
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
            description="Returns a source file's symbol tree.",
            parameters=DocumentSymbolsInput,
            manager=manager,
            guidance="Use to explore a large file's structure before reading selected sections.",
            prompt=DOCUMENT_SYMBOLS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path") or "")
        try:
            result = self.codeintel_manager.get_document_symbols(
                file_path=file_path,
            )
            return self._success_result(action="get_document_symbols", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"document symbols 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
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
            description="Searches workspace symbols by name.",
            parameters=WorkspaceSymbolsInput,
            manager=manager,
            guidance="Use a stable name fragment, not a natural-language sentence.",
            prompt=WORKSPACE_SYMBOLS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        try:
            result = self.codeintel_manager.get_workspace_symbols(
                query=normalize_generic_input(parameters.get("query") or ""),
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
            description="Reads a file's language-server diagnostics.",
            parameters=DiagnosticsInput,
            manager=manager,
            guidance="Empty diagnostics do not prove the file is correct.",
            prompt=DIAGNOSTICS_PROMPT,
        )

    def run(self, parameters: dict) -> ToolResult:
        file_path = normalize_path_input(parameters.get("file_path") or "")
        try:
            result = self.codeintel_manager.get_diagnostics(
                file_path=file_path,
            )
            return self._success_result(action="get_diagnostics", payload=result.to_dict())
        except PathResolutionError as exc:
            return self._tool_error(
                f"diagnostics 查询失败: {exc}",
                error_type="invalid_path",
                metadata={"file_path": file_path},
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
            description="Reports code intelligence cache and offline index status.",
            parameters=CodeIntelCacheStatusInput,
            manager=manager,
            guidance="Inspect the cache before deciding whether to prewarm the workspace.",
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
            description="Prewarms the code intelligence cache and offline symbol index.",
            parameters=CodeIntelPrewarmWorkspaceInput,
            manager=manager,
            guidance="Bound work with path_prefix and max_files.",
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
    expose_in_deferred: bool | None = True,
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
    registry.register_tool(CodeIntelStatusTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(FindDefinitionTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(FindReferencesTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(GetDocumentSymbolsTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(GetWorkspaceSymbolsTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(GetDiagnosticsTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(CodeIntelCacheStatusTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
    registry.register_tool(CodeIntelPrewarmWorkspaceTool(codeintel_manager), expose_in_deferred=expose_in_deferred)
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
