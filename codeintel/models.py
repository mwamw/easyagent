"""Shared models for code intelligence providers and tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Optional


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(slots=True)
class CodePosition:
    line: int
    character: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CodePosition":
        data = dict(payload or {})
        return cls(
            line=int(data.get("line", 0)),
            character=int(data.get("character", 0)),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "line": int(self.line),
            "character": int(self.character),
        }


@dataclass(slots=True)
class CodeRange:
    start: CodePosition
    end: CodePosition

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CodeRange":
        data = dict(payload or {})
        return cls(
            start=CodePosition.from_dict(data.get("start")),
            end=CodePosition.from_dict(data.get("end")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }


@dataclass(slots=True)
class CodeLocation:
    path: str
    uri: str
    range: CodeRange
    name: Optional[str] = None
    kind: Optional[str] = None
    detail: Optional[str] = None
    preview: Optional[str] = None
    container_name: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CodeLocation":
        data = dict(payload or {})
        return cls(
            path=str(data.get("path") or ""),
            uri=str(data.get("uri") or ""),
            range=CodeRange.from_dict(data.get("range")),
            name=str(data.get("name") or "") or None,
            kind=str(data.get("kind") or "") or None,
            detail=str(data.get("detail") or "") or None,
            preview=str(data.get("preview") or "") or None,
            container_name=str(data.get("containerName") or data.get("container_name") or "") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "uri": self.uri,
            "range": self.range.to_dict(),
        }
        if self.name:
            payload["name"] = self.name
        if self.kind:
            payload["kind"] = self.kind
        if self.detail:
            payload["detail"] = self.detail
        if self.preview:
            payload["preview"] = self.preview
        if self.container_name:
            payload["containerName"] = self.container_name
        return payload


@dataclass(slots=True)
class SymbolRecord:
    name: str
    kind: str
    path: str
    uri: str
    range: CodeRange
    selection_range: Optional[CodeRange] = None
    detail: Optional[str] = None
    container_name: Optional[str] = None
    children: list["SymbolRecord"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SymbolRecord":
        data = dict(payload or {})
        return cls(
            name=str(data.get("name") or ""),
            kind=str(data.get("kind") or ""),
            path=str(data.get("path") or ""),
            uri=str(data.get("uri") or ""),
            range=CodeRange.from_dict(data.get("range")),
            selection_range=(
                CodeRange.from_dict(data.get("selectionRange") or data.get("selection_range"))
                if (data.get("selectionRange") or data.get("selection_range")) is not None
                else None
            ),
            detail=str(data.get("detail") or "") or None,
            container_name=str(data.get("containerName") or data.get("container_name") or "") or None,
            children=[cls.from_dict(item) for item in list(data.get("children") or []) if item is not None],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "uri": self.uri,
            "range": self.range.to_dict(),
            "children": [child.to_dict() for child in self.children],
        }
        if self.selection_range is not None:
            payload["selectionRange"] = self.selection_range.to_dict()
        if self.detail:
            payload["detail"] = self.detail
        if self.container_name:
            payload["containerName"] = self.container_name
        return payload


@dataclass(slots=True)
class DiagnosticRecord:
    path: str
    uri: str
    severity: str
    message: str
    range: CodeRange
    code: Optional[str] = None
    source: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DiagnosticRecord":
        data = dict(payload or {})
        return cls(
            path=str(data.get("path") or ""),
            uri=str(data.get("uri") or ""),
            severity=str(data.get("severity") or ""),
            message=str(data.get("message") or ""),
            range=CodeRange.from_dict(data.get("range")),
            code=str(data.get("code") or "") or None,
            source=str(data.get("source") or "") or None,
            tags=[str(item) for item in list(data.get("tags") or []) if item is not None],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "uri": self.uri,
            "severity": self.severity,
            "message": self.message,
            "range": self.range.to_dict(),
            "tags": list(self.tags),
        }
        if self.code:
            payload["code"] = self.code
        if self.source:
            payload["source"] = self.source
        return payload


@dataclass(slots=True)
class CodeIntelAvailability:
    available: bool
    provider_name: str
    workspace_root: str
    server_command: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CodeIntelAvailability":
        data = dict(payload or {})
        return cls(
            available=bool(data.get("available")),
            provider_name=str(data.get("providerName") or data.get("provider_name") or "codeintel"),
            workspace_root=str(data.get("workspaceRoot") or data.get("workspace_root") or ""),
            server_command=[str(item) for item in list(data.get("serverCommand") or data.get("server_command") or []) if item],
            reason=str(data.get("reason") or "") or None,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "available": bool(self.available),
            "providerName": self.provider_name,
            "workspaceRoot": self.workspace_root,
            "serverCommand": list(self.server_command),
            "metadata": dict(self.metadata),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class CodeIntelQueryResult:
    status: str
    provider_name: str
    workspace_root: str
    items: list[Any] = field(default_factory=list)
    fallback_tools: list[str] = field(default_factory=lambda: ["FileRead", "Grep", "Glob"])
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        *,
        item_loader: Any | None = None,
    ) -> "CodeIntelQueryResult":
        data = dict(payload or {})
        raw_items = list(data.get("items") or [])
        if callable(item_loader):
            items = [item_loader(item) for item in raw_items if item is not None]
        else:
            items = raw_items
        return cls(
            status=str(data.get("status") or "unknown"),
            provider_name=str(data.get("providerName") or data.get("provider_name") or "codeintel"),
            workspace_root=str(data.get("workspaceRoot") or data.get("workspace_root") or ""),
            items=items,
            fallback_tools=[str(item) for item in list(data.get("fallbackTools") or data.get("fallback_tools") or ["FileRead", "Grep", "Glob"]) if item],
            error_message=str(data.get("errorMessage") or data.get("error_message") or "") or None,
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def ok(
        cls,
        *,
        provider_name: str,
        workspace_root: str,
        items: Optional[list[Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "CodeIntelQueryResult":
        return cls(
            status="ok",
            provider_name=provider_name,
            workspace_root=workspace_root,
            items=list(items or []),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def unavailable(
        cls,
        *,
        provider_name: str,
        workspace_root: str,
        error_message: str,
        metadata: Optional[dict[str, Any]] = None,
        fallback_tools: Optional[list[str]] = None,
    ) -> "CodeIntelQueryResult":
        return cls(
            status="unavailable",
            provider_name=provider_name,
            workspace_root=workspace_root,
            error_message=error_message,
            metadata=dict(metadata or {}),
            fallback_tools=list(fallback_tools or ["FileRead", "Grep", "Glob"]),
        )

    @classmethod
    def error(
        cls,
        *,
        provider_name: str,
        workspace_root: str,
        error_message: str,
        metadata: Optional[dict[str, Any]] = None,
        fallback_tools: Optional[list[str]] = None,
    ) -> "CodeIntelQueryResult":
        return cls(
            status="error",
            provider_name=provider_name,
            workspace_root=workspace_root,
            error_message=error_message,
            metadata=dict(metadata or {}),
            fallback_tools=list(fallback_tools or ["FileRead", "Grep", "Glob"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "providerName": self.provider_name,
            "workspaceRoot": self.workspace_root,
            "items": [_json_safe(item) for item in self.items],
            "fallbackTools": list(self.fallback_tools),
            "metadata": _json_safe(self.metadata),
        }
        if self.error_message:
            payload["errorMessage"] = self.error_message
        return payload


@dataclass(slots=True)
class DefinitionQuery:
    workspace_root: str
    file_path: str
    line: int
    column: int


@dataclass(slots=True)
class ReferenceQuery:
    workspace_root: str
    file_path: str
    line: int
    column: int
    include_declaration: bool = False


@dataclass(slots=True)
class DocumentSymbolsQuery:
    workspace_root: str
    file_path: str


@dataclass(slots=True)
class WorkspaceSymbolsQuery:
    workspace_root: str
    query: str
    limit: int = 50


@dataclass(slots=True)
class DiagnosticsQuery:
    workspace_root: str
    file_path: str


__all__ = [
    "CodeIntelAvailability",
    "CodeIntelQueryResult",
    "CodeLocation",
    "CodePosition",
    "CodeRange",
    "DefinitionQuery",
    "DiagnosticRecord",
    "DiagnosticsQuery",
    "DocumentSymbolsQuery",
    "ReferenceQuery",
    "SymbolRecord",
    "WorkspaceSymbolsQuery",
]
