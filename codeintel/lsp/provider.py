"""LSP-backed code intelligence provider."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from ..models import (
    CodeIntelAvailability,
    CodeIntelQueryResult,
    CodeLocation,
    CodePosition,
    CodeRange,
    DefinitionQuery,
    DiagnosticRecord,
    DiagnosticsQuery,
    DocumentSymbolsQuery,
    ReferenceQuery,
    SymbolRecord,
    WorkspaceSymbolsQuery,
)
from ..provider import CodeIntelProvider, DEFAULT_FALLBACK_TOOLS
from .client import LSPClient, LSPClientError, uri_to_path


SYMBOL_KIND_NAMES = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}

DIAGNOSTIC_SEVERITY_NAMES = {
    1: "error",
    2: "warning",
    3: "information",
    4: "hint",
}

DEFAULT_SERVER_CANDIDATES: dict[str, list[list[str]]] = {
    "python": [
        ["basedpyright-langserver", "--stdio"],
        ["pyright-langserver", "--stdio"],
        ["pylsp"],
        ["jedi-language-server"],
    ],
    "typescript": [
        ["typescript-language-server", "--stdio"],
    ],
    "javascript": [
        ["typescript-language-server", "--stdio"],
    ],
    "rust": [
        ["rust-analyzer"],
    ],
    "go": [
        ["gopls"],
    ],
    "java": [
        ["jdtls"],
    ],
    "c": [
        ["clangd"],
    ],
    "cpp": [
        ["clangd"],
    ],
}


def _language_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mapping = {
        ".py": "python",
        ".pyi": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".hpp": "cpp",
    }
    return mapping.get(suffix, "plaintext")


def _guess_workspace_anchor_file(workspace_root: str) -> Optional[str]:
    preferred_suffixes = {
        ".py",
        ".pyi",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cc",
        ".cpp",
        ".hpp",
        ".h",
    }
    for root, dirs, files in os.walk(workspace_root):
        dirs[:] = [item for item in dirs if item not in {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}]
        for file_name in files:
            suffix = Path(file_name).suffix.lower()
            if suffix in preferred_suffixes:
                return os.path.join(root, file_name)
    return None


def _normalize_server_command(command: list[str]) -> list[str]:
    if not command:
        return []
    binary = command[0]
    resolved = shutil.which(binary) if not os.path.isabs(binary) else (binary if os.path.exists(binary) else None)
    if resolved is None:
        return []
    return [resolved, *command[1:]]


def _preview_for_location(path: str, code_range: CodeRange) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except Exception:
        return None
    start_line = max(1, int(code_range.start.line))
    if start_line > len(lines):
        return None
    return lines[start_line - 1].strip()


class LSPCodeIntelProvider(CodeIntelProvider):
    provider_name = "lsp"

    def __init__(
        self,
        *,
        server_command: Optional[list[str]] = None,
        server_env: Optional[dict[str, str]] = None,
        request_timeout_ms: int = 5000,
        diagnostics_wait_ms: int = 800,
    ):
        self.server_command = list(server_command or [])
        self.server_env = dict(server_env or {})
        self.request_timeout_ms = max(100, int(request_timeout_ms))
        self.diagnostics_wait_ms = max(100, int(diagnostics_wait_ms))
        self._lock = threading.RLock()
        self._sessions: dict[tuple[str, tuple[str, ...]], LSPClient] = {}

    def export_state(self) -> dict[str, Any]:
        return {
            "kind": "lsp",
            "providerName": self.provider_name,
            "serverCommand": list(self.server_command),
            "serverEnv": dict(self.server_env),
            "requestTimeoutMs": int(self.request_timeout_ms),
            "diagnosticsWaitMs": int(self.diagnostics_wait_ms),
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any] | None) -> "LSPCodeIntelProvider":
        data = dict(payload or {})
        return cls(
            server_command=list(data.get("serverCommand") or []),
            server_env=dict(data.get("serverEnv") or {}),
            request_timeout_ms=int(data.get("requestTimeoutMs") or 5000),
            diagnostics_wait_ms=int(data.get("diagnosticsWaitMs") or 800),
        )

    def _resolve_server_command(self, *, file_path: Optional[str]) -> tuple[list[str], Optional[str], str]:
        language = _language_from_path(file_path or "") if file_path else "plaintext"
        if self.server_command:
            command = _normalize_server_command(self.server_command)
            if command:
                return command, None, language
            return [], f"显式配置的 LSP server 不存在或不可执行: {self.server_command}", language
        candidates = DEFAULT_SERVER_CANDIDATES.get(language, [])
        for candidate in candidates:
            command = _normalize_server_command(candidate)
            if command:
                return command, None, language
        if not candidates:
            return [], f"当前文件类型暂未配置默认 LSP server: {language}", language
        return [], f"未找到可用的 {language} LSP server，可考虑安装: {candidates}", language

    def get_status(
        self,
        *,
        workspace_root: str,
        file_path: Optional[str] = None,
    ) -> CodeIntelAvailability:
        anchor_path = file_path or _guess_workspace_anchor_file(workspace_root)
        command, reason, language = self._resolve_server_command(file_path=anchor_path)
        return CodeIntelAvailability(
            available=bool(command),
            provider_name=self.provider_name,
            workspace_root=os.path.abspath(workspace_root),
            server_command=command,
            reason=reason,
            metadata={"language": language, "anchorPath": anchor_path},
        )

    def _get_session(
        self,
        *,
        workspace_root: str,
        file_path: Optional[str],
    ) -> tuple[Optional[LSPClient], Optional[str], dict[str, Any]]:
        status = self.get_status(workspace_root=workspace_root, file_path=file_path)
        metadata = status.to_dict()
        if not status.available:
            return None, status.reason or "LSP 不可用。", metadata
        key = (os.path.abspath(workspace_root), tuple(status.server_command))
        with self._lock:
            client = self._sessions.get(key)
            if client is None:
                try:
                    client = LSPClient(
                        server_command=status.server_command,
                        workspace_root=workspace_root,
                        env=self.server_env,
                        request_timeout_ms=self.request_timeout_ms,
                        diagnostics_wait_ms=self.diagnostics_wait_ms,
                    )
                except Exception as exc:
                    return None, f"启动 LSP server 失败: {exc}", metadata
                self._sessions[key] = client
        return client, None, metadata

    @staticmethod
    def _to_code_range(payload: dict[str, Any]) -> CodeRange:
        start = dict(payload.get("start") or {})
        end = dict(payload.get("end") or {})
        return CodeRange(
            start=CodePosition(
                line=int(start.get("line", 0)) + 1,
                character=int(start.get("character", 0)) + 1,
            ),
            end=CodePosition(
                line=int(end.get("line", 0)) + 1,
                character=int(end.get("character", 0)) + 1,
            ),
        )

    def _to_location(self, payload: dict[str, Any]) -> CodeLocation:
        location_payload = dict(payload or {})
        uri = str(location_payload.get("uri") or location_payload.get("targetUri") or "")
        code_range_payload = location_payload.get("range") or location_payload.get("targetRange") or {}
        path = uri_to_path(uri)
        code_range = self._to_code_range(dict(code_range_payload))
        return CodeLocation(
            path=path,
            uri=uri,
            range=code_range,
            preview=_preview_for_location(path, code_range),
        )

    def _location_result(
        self,
        *,
        workspace_root: str,
        file_path: str,
        method_name: str,
        resolver: Any,
    ) -> CodeIntelQueryResult:
        client, unavailable_reason, metadata = self._get_session(
            workspace_root=workspace_root,
            file_path=file_path,
        )
        if client is None:
            return CodeIntelQueryResult.unavailable(
                provider_name=self.provider_name,
                workspace_root=workspace_root,
                error_message=unavailable_reason or "LSP 不可用。",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )
        try:
            raw_locations = resolver(client)
            items = [self._to_location(payload) for payload in raw_locations]
            return CodeIntelQueryResult.ok(
                provider_name=self.provider_name,
                workspace_root=workspace_root,
                items=items,
                metadata={
                    **metadata,
                    "method": method_name,
                    "resultCount": len(items),
                },
            )
        except (LSPClientError, ValueError) as exc:
            return CodeIntelQueryResult.error(
                provider_name=self.provider_name,
                workspace_root=workspace_root,
                error_message=f"{method_name} 查询失败: {exc}",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )

    def find_definition(self, query: DefinitionQuery) -> CodeIntelQueryResult:
        return self._location_result(
            workspace_root=query.workspace_root,
            file_path=query.file_path,
            method_name="definition",
            resolver=lambda client: client.definition(
                query.file_path,
                line_zero_based=max(0, int(query.line) - 1),
                column_zero_based=max(0, int(query.column) - 1),
            ),
        )

    def find_references(self, query: ReferenceQuery) -> CodeIntelQueryResult:
        return self._location_result(
            workspace_root=query.workspace_root,
            file_path=query.file_path,
            method_name="references",
            resolver=lambda client: client.references(
                query.file_path,
                line_zero_based=max(0, int(query.line) - 1),
                column_zero_based=max(0, int(query.column) - 1),
                include_declaration=query.include_declaration,
            ),
        )

    def _document_symbol_from_payload(self, payload: dict[str, Any], *, fallback_path: str, fallback_uri: str) -> SymbolRecord:
        symbol_payload = dict(payload or {})
        location = symbol_payload.get("location")
        if location:
            location_payload = dict(location or {})
            uri = str(location_payload.get("uri") or fallback_uri)
            path = uri_to_path(uri)
            code_range = self._to_code_range(dict(location_payload.get("range") or {}))
            selection_range = code_range
        else:
            uri = fallback_uri
            path = fallback_path
            code_range = self._to_code_range(dict(symbol_payload.get("range") or {}))
            selection_payload = symbol_payload.get("selectionRange")
            selection_range = self._to_code_range(dict(selection_payload or {})) if selection_payload else code_range

        return SymbolRecord(
            name=str(symbol_payload.get("name") or ""),
            kind=SYMBOL_KIND_NAMES.get(int(symbol_payload.get("kind") or 0), str(symbol_payload.get("kind") or "Unknown")),
            path=path,
            uri=uri,
            range=code_range,
            selection_range=selection_range,
            detail=str(symbol_payload.get("detail") or "") or None,
            container_name=str(symbol_payload.get("containerName") or "") or None,
            children=[
                self._document_symbol_from_payload(item, fallback_path=fallback_path, fallback_uri=fallback_uri)
                for item in list(symbol_payload.get("children") or [])
                if item
            ],
        )

    def get_document_symbols(self, query: DocumentSymbolsQuery) -> CodeIntelQueryResult:
        client, unavailable_reason, metadata = self._get_session(
            workspace_root=query.workspace_root,
            file_path=query.file_path,
        )
        if client is None:
            return CodeIntelQueryResult.unavailable(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                error_message=unavailable_reason or "LSP 不可用。",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )
        try:
            raw_items = client.document_symbols(query.file_path)
            uri = Path(query.file_path).as_uri()
            items = [
                self._document_symbol_from_payload(item, fallback_path=query.file_path, fallback_uri=uri)
                for item in raw_items
            ]
            return CodeIntelQueryResult.ok(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                items=items,
                metadata={
                    **metadata,
                    "method": "documentSymbol",
                    "resultCount": len(items),
                },
            )
        except (LSPClientError, ValueError) as exc:
            return CodeIntelQueryResult.error(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                error_message=f"documentSymbol 查询失败: {exc}",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )

    def get_workspace_symbols(self, query: WorkspaceSymbolsQuery) -> CodeIntelQueryResult:
        client, unavailable_reason, metadata = self._get_session(
            workspace_root=query.workspace_root,
            file_path=_guess_workspace_anchor_file(query.workspace_root),
        )
        if client is None:
            return CodeIntelQueryResult.unavailable(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                error_message=unavailable_reason or "LSP 不可用。",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )
        try:
            raw_items = client.workspace_symbols(query.query)
            symbols = [
                self._document_symbol_from_payload(
                    item,
                    fallback_path=uri_to_path(str(dict(item.get("location") or {}).get("uri") or Path(query.workspace_root).as_uri())),
                    fallback_uri=str(dict(item.get("location") or {}).get("uri") or Path(query.workspace_root).as_uri()),
                )
                for item in raw_items[: max(1, int(query.limit))]
            ]
            return CodeIntelQueryResult.ok(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                items=symbols,
                metadata={
                    **metadata,
                    "method": "workspace/symbol",
                    "query": query.query,
                    "resultCount": len(symbols),
                    "limit": int(query.limit),
                },
            )
        except (LSPClientError, ValueError) as exc:
            return CodeIntelQueryResult.error(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                error_message=f"workspace/symbol 查询失败: {exc}",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )

    def get_diagnostics(self, query: DiagnosticsQuery) -> CodeIntelQueryResult:
        client, unavailable_reason, metadata = self._get_session(
            workspace_root=query.workspace_root,
            file_path=query.file_path,
        )
        if client is None:
            return CodeIntelQueryResult.unavailable(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                error_message=unavailable_reason or "LSP 不可用。",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )
        try:
            raw_items = client.wait_for_diagnostics(query.file_path)
            uri = Path(query.file_path).as_uri()
            diagnostics = [
                DiagnosticRecord(
                    path=query.file_path,
                    uri=uri,
                    severity=DIAGNOSTIC_SEVERITY_NAMES.get(int(dict(item).get("severity") or 3), "information"),
                    message=str(dict(item).get("message") or ""),
                    range=self._to_code_range(dict(dict(item).get("range") or {})),
                    code=str(dict(item).get("code") or "") or None,
                    source=str(dict(item).get("source") or "") or None,
                    tags=[str(tag) for tag in list(dict(item).get("tags") or []) if tag is not None],
                )
                for item in raw_items
            ]
            return CodeIntelQueryResult.ok(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                items=diagnostics,
                metadata={
                    **metadata,
                    "method": "publishDiagnostics",
                    "resultCount": len(diagnostics),
                },
            )
        except (LSPClientError, ValueError) as exc:
            return CodeIntelQueryResult.error(
                provider_name=self.provider_name,
                workspace_root=query.workspace_root,
                error_message=f"diagnostics 查询失败: {exc}",
                metadata=metadata,
                fallback_tools=list(DEFAULT_FALLBACK_TOOLS),
            )

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception:
                continue


__all__ = ["LSPCodeIntelProvider"]
