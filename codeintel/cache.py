"""Workspace-aware code intelligence cache and offline symbol snapshot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    CodeIntelAvailability,
    CodeIntelQueryResult,
    CodeLocation,
    DiagnosticRecord,
    SymbolRecord,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: str) -> str:
    return os.path.abspath(str(path or ""))


def _safe_mtime_ns(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _restore_query_result(action: str, payload: dict[str, Any]) -> CodeIntelQueryResult:
    item_loader = None
    if action in {"definition", "references"}:
        item_loader = CodeLocation.from_dict
    elif action in {"document_symbols", "workspace_symbols"}:
        item_loader = SymbolRecord.from_dict
    elif action == "diagnostics":
        item_loader = DiagnosticRecord.from_dict
    return CodeIntelQueryResult.from_dict(payload, item_loader=item_loader)


def _clone_result(result: CodeIntelQueryResult, *, metadata_updates: Optional[dict[str, Any]] = None) -> CodeIntelQueryResult:
    metadata = dict(result.metadata or {})
    metadata.update(dict(metadata_updates or {}))
    return CodeIntelQueryResult(
        status=result.status,
        provider_name=result.provider_name,
        workspace_root=result.workspace_root,
        items=list(result.items or []),
        fallback_tools=list(result.fallback_tools or []),
        error_message=result.error_message,
        metadata=metadata,
    )


@dataclass(slots=True)
class CachedFileEntry:
    path: str
    mtime_ns: Optional[int] = None
    document_symbols_payload: Optional[dict[str, Any]] = None
    diagnostics_payload: Optional[dict[str, Any]] = None
    document_symbols_updated_at: Optional[str] = None
    diagnostics_updated_at: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CachedFileEntry":
        data = dict(payload or {})
        return cls(
            path=_normalize_path(str(data.get("path") or "")),
            mtime_ns=(int(data["mtimeNs"]) if data.get("mtimeNs") is not None else None),
            document_symbols_payload=dict(data.get("documentSymbols") or {}) or None,
            diagnostics_payload=dict(data.get("diagnostics") or {}) or None,
            document_symbols_updated_at=str(data.get("documentSymbolsUpdatedAt") or "") or None,
            diagnostics_updated_at=str(data.get("diagnosticsUpdatedAt") or "") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "mtimeNs": self.mtime_ns,
        }
        if self.document_symbols_payload is not None:
            payload["documentSymbols"] = dict(self.document_symbols_payload)
        if self.diagnostics_payload is not None:
            payload["diagnostics"] = dict(self.diagnostics_payload)
        if self.document_symbols_updated_at:
            payload["documentSymbolsUpdatedAt"] = self.document_symbols_updated_at
        if self.diagnostics_updated_at:
            payload["diagnosticsUpdatedAt"] = self.diagnostics_updated_at
        return payload


@dataclass(slots=True)
class CachedQueryEntry:
    action: str
    query_key: str
    fingerprint: str
    payload: dict[str, Any]
    updated_at: str = field(default_factory=_utc_now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CachedQueryEntry":
        data = dict(payload or {})
        return cls(
            action=str(data.get("action") or ""),
            query_key=str(data.get("queryKey") or ""),
            fingerprint=str(data.get("fingerprint") or ""),
            payload=dict(data.get("payload") or {}),
            updated_at=str(data.get("updatedAt") or _utc_now_iso()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "queryKey": self.query_key,
            "fingerprint": self.fingerprint,
            "payload": dict(self.payload),
            "updatedAt": self.updated_at,
        }


class WorkspaceCodeIntelCache:
    """Keeps per-workspace codeintel snapshots that survive provider restarts."""

    def __init__(self, workspace_root: str):
        self.workspace_root = _normalize_path(workspace_root)
        self.last_status: Optional[CodeIntelAvailability] = None
        self.file_entries: dict[str, CachedFileEntry] = {}
        self.query_entries: dict[str, CachedQueryEntry] = {}
        self.last_prewarm_at: Optional[str] = None
        self.last_prewarm_summary: dict[str, Any] = {}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "WorkspaceCodeIntelCache":
        data = dict(payload or {})
        cache = cls(str(data.get("workspaceRoot") or os.getcwd()))
        status_payload = data.get("lastStatus")
        if status_payload:
            cache.last_status = CodeIntelAvailability.from_dict(dict(status_payload))
        cache.file_entries = {
            _normalize_path(item.get("path") or key): CachedFileEntry.from_dict(item)
            for key, item in dict(data.get("fileEntries") or {}).items()
            if item
        }
        cache.query_entries = {
            f"{entry.action}:{entry.query_key}": entry
            for entry in (
                CachedQueryEntry.from_dict(item)
                for item in list(data.get("queryEntries") or [])
                if item
            )
        }
        cache.last_prewarm_at = str(data.get("lastPrewarmAt") or "") or None
        cache.last_prewarm_summary = dict(data.get("lastPrewarmSummary") or {})
        return cache

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "workspaceRoot": self.workspace_root,
            "fileEntries": {
                path: entry.to_dict()
                for path, entry in self.file_entries.items()
            },
            "queryEntries": [entry.to_dict() for entry in self.query_entries.values()],
            "lastPrewarmSummary": dict(self.last_prewarm_summary),
        }
        if self.last_status is not None:
            payload["lastStatus"] = self.last_status.to_dict()
        if self.last_prewarm_at:
            payload["lastPrewarmAt"] = self.last_prewarm_at
        return payload

    def record_status(self, status: CodeIntelAvailability) -> None:
        self.last_status = status

    def record_document_symbols(self, path: str, result: CodeIntelQueryResult) -> None:
        normalized = _normalize_path(path)
        entry = self.file_entries.get(normalized) or CachedFileEntry(path=normalized)
        entry.mtime_ns = _safe_mtime_ns(normalized)
        entry.document_symbols_payload = result.to_dict()
        entry.document_symbols_updated_at = _utc_now_iso()
        self.file_entries[normalized] = entry

    def record_diagnostics(self, path: str, result: CodeIntelQueryResult) -> None:
        normalized = _normalize_path(path)
        entry = self.file_entries.get(normalized) or CachedFileEntry(path=normalized)
        entry.mtime_ns = _safe_mtime_ns(normalized)
        entry.diagnostics_payload = result.to_dict()
        entry.diagnostics_updated_at = _utc_now_iso()
        self.file_entries[normalized] = entry

    def record_query(
        self,
        *,
        action: str,
        query_key: str,
        fingerprint: str,
        result: CodeIntelQueryResult,
    ) -> None:
        entry = CachedQueryEntry(
            action=action,
            query_key=str(query_key),
            fingerprint=str(fingerprint),
            payload=result.to_dict(),
        )
        self.query_entries[f"{entry.action}:{entry.query_key}"] = entry

    def get_cached_document_symbols(self, path: str) -> tuple[CodeIntelQueryResult, dict[str, Any]] | None:
        normalized = _normalize_path(path)
        entry = self.file_entries.get(normalized)
        if entry is None or entry.document_symbols_payload is None:
            return None
        result = _restore_query_result("document_symbols", entry.document_symbols_payload)
        current_mtime_ns = _safe_mtime_ns(normalized)
        stale = entry.mtime_ns is not None and current_mtime_ns is not None and entry.mtime_ns != current_mtime_ns
        return _clone_result(
            result,
            metadata_updates={
                "cacheHit": True,
                "cacheSource": "document_symbols",
                "cacheStale": bool(stale),
                "cachedAt": entry.document_symbols_updated_at,
                "cachedFileMtimeNs": entry.mtime_ns,
                "currentFileMtimeNs": current_mtime_ns,
            },
        ), {
            "fresh": not stale,
            "stale": bool(stale),
            "cachedAt": entry.document_symbols_updated_at,
        }

    def get_cached_diagnostics(self, path: str) -> tuple[CodeIntelQueryResult, dict[str, Any]] | None:
        normalized = _normalize_path(path)
        entry = self.file_entries.get(normalized)
        if entry is None or entry.diagnostics_payload is None:
            return None
        result = _restore_query_result("diagnostics", entry.diagnostics_payload)
        current_mtime_ns = _safe_mtime_ns(normalized)
        stale = entry.mtime_ns is not None and current_mtime_ns is not None and entry.mtime_ns != current_mtime_ns
        return _clone_result(
            result,
            metadata_updates={
                "cacheHit": True,
                "cacheSource": "diagnostics",
                "cacheStale": bool(stale),
                "cachedAt": entry.diagnostics_updated_at,
                "cachedFileMtimeNs": entry.mtime_ns,
                "currentFileMtimeNs": current_mtime_ns,
            },
        ), {
            "fresh": not stale,
            "stale": bool(stale),
            "cachedAt": entry.diagnostics_updated_at,
        }

    def get_cached_query(
        self,
        *,
        action: str,
        query_key: str,
        fingerprint: str,
    ) -> tuple[CodeIntelQueryResult, dict[str, Any]] | None:
        entry = self.query_entries.get(f"{action}:{query_key}")
        if entry is None:
            return None
        result = _restore_query_result(action, entry.payload)
        stale = str(entry.fingerprint) != str(fingerprint)
        return _clone_result(
            result,
            metadata_updates={
                "cacheHit": True,
                "cacheSource": action,
                "cacheStale": bool(stale),
                "cachedAt": entry.updated_at,
                "queryKey": query_key,
            },
        ), {
            "fresh": not stale,
            "stale": bool(stale),
            "cachedAt": entry.updated_at,
        }

    def search_symbols(
        self,
        *,
        query: str,
        limit: int,
        provider_name: str,
    ) -> CodeIntelQueryResult | None:
        normalized_query = str(query or "").strip().casefold()
        matches: list[SymbolRecord] = []
        indexed_files = 0

        def visit(symbol: SymbolRecord) -> None:
            haystacks = [
                symbol.name,
                symbol.detail or "",
                symbol.container_name or "",
            ]
            if not normalized_query or any(normalized_query in value.casefold() for value in haystacks if value):
                matches.append(symbol)
            for child in symbol.children:
                visit(child)

        for entry in self.file_entries.values():
            if not entry.document_symbols_payload:
                continue
            indexed_files += 1
            result = _restore_query_result("document_symbols", entry.document_symbols_payload)
            for item in result.items:
                if isinstance(item, SymbolRecord):
                    visit(item)
        if not matches:
            return None
        return CodeIntelQueryResult.ok(
            provider_name=provider_name,
            workspace_root=self.workspace_root,
            items=matches[: max(1, int(limit))],
            metadata={
                "cacheHit": True,
                "cacheSource": "offline_index",
                "offlineIndexAvailable": True,
                "query": str(query or ""),
                "limit": max(1, int(limit)),
                "indexedFileCount": indexed_files,
                "resultCount": min(len(matches), max(1, int(limit))),
                "lastPrewarmAt": self.last_prewarm_at,
            },
        )

    def mark_prewarm(self, summary: dict[str, Any]) -> None:
        self.last_prewarm_at = _utc_now_iso()
        self.last_prewarm_summary = dict(summary or {})

    def get_status_payload(self) -> dict[str, Any]:
        indexed_file_count = sum(1 for entry in self.file_entries.values() if entry.document_symbols_payload)
        diagnostics_file_count = sum(1 for entry in self.file_entries.values() if entry.diagnostics_payload)
        symbol_count = 0
        for entry in self.file_entries.values():
            if not entry.document_symbols_payload:
                continue
            result = _restore_query_result("document_symbols", entry.document_symbols_payload)
            symbol_count += len(result.items)
        return {
            "workspaceRoot": self.workspace_root,
            "indexedFileCount": indexed_file_count,
            "diagnosticsFileCount": diagnostics_file_count,
            "cachedQueryCount": len(self.query_entries),
            "symbolCount": symbol_count,
            "offlineIndexAvailable": indexed_file_count > 0,
            "lastPrewarmAt": self.last_prewarm_at,
            "lastPrewarmSummary": dict(self.last_prewarm_summary),
            "lastStatus": self.last_status.to_dict() if self.last_status is not None else None,
        }


__all__ = [
    "CachedFileEntry",
    "CachedQueryEntry",
    "WorkspaceCodeIntelCache",
]
