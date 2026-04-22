"""Runtime manager for workspace-aware code intelligence access."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from Tool.runtime import PathResolver, PathResolutionError

from .cache import WorkspaceCodeIntelCache
from .models import (
    CodeIntelAvailability,
    CodeIntelQueryResult,
    DefinitionQuery,
    DiagnosticsQuery,
    DocumentSymbolsQuery,
    ReferenceQuery,
    WorkspaceSymbolsQuery,
)
from .provider import CodeIntelProvider


DEFAULT_CODEINTEL_SUFFIXES = (
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
)

DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}


class CodeIntelManager:
    def __init__(
        self,
        *,
        provider: CodeIntelProvider,
        parent_agent: Any | None = None,
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        cache: WorkspaceCodeIntelCache | None = None,
        workspace_caches: Optional[Iterable[WorkspaceCodeIntelCache]] = None,
    ):
        self.provider = provider
        self.parent_agent = parent_agent
        self.workspace_root = os.path.abspath(workspace_root or self._infer_workspace_root(parent_agent) or os.getcwd())
        inferred_roots = tuple(
            os.path.abspath(item)
            for item in (allowed_roots or self._infer_allowed_roots(parent_agent))
            if item
        )
        self.allowed_roots = inferred_roots or (self.workspace_root,)
        self.last_status: Optional[CodeIntelAvailability] = None
        self._workspace_caches: dict[str, WorkspaceCodeIntelCache] = {}
        for item in list(workspace_caches or []):
            if item is None:
                continue
            self._workspace_caches[os.path.abspath(item.workspace_root)] = item
        if cache is not None:
            self._workspace_caches[os.path.abspath(cache.workspace_root)] = cache
        if self.workspace_root not in self._workspace_caches:
            self._workspace_caches[self.workspace_root] = WorkspaceCodeIntelCache(self.workspace_root)

    @staticmethod
    def _infer_workspace_root(parent_agent: Any | None) -> Optional[str]:
        if parent_agent is None:
            return None
        execution_context = getattr(parent_agent, "execution_context", None)
        worktree_path = getattr(execution_context, "worktree_path", None)
        if worktree_path:
            return os.path.abspath(str(worktree_path))
        workspace_root = getattr(execution_context, "workspace_root", None)
        if workspace_root:
            return os.path.abspath(str(workspace_root))
        config = getattr(parent_agent, "config", None)
        workspace_root = getattr(config, "workspace_root", None)
        if workspace_root:
            return os.path.abspath(str(workspace_root))
        return None

    @staticmethod
    def _infer_allowed_roots(parent_agent: Any | None) -> tuple[str, ...]:
        if parent_agent is None:
            return ()
        execution_context = getattr(parent_agent, "execution_context", None)
        roots = tuple(getattr(execution_context, "allowed_roots", ()) or ())
        if roots:
            return tuple(os.path.abspath(str(item)) for item in roots if item)
        config = getattr(parent_agent, "config", None)
        if config is not None and hasattr(config, "get_allowed_roots"):
            try:
                values = config.get_allowed_roots()
            except Exception:
                values = []
            return tuple(os.path.abspath(str(item)) for item in values if item)
        return ()

    def bind_parent_agent(self, parent_agent: Any | None) -> None:
        self.parent_agent = parent_agent

    def _cache_for(self, workspace_root: str) -> WorkspaceCodeIntelCache:
        normalized = os.path.abspath(workspace_root)
        cache = self._workspace_caches.get(normalized)
        if cache is None:
            cache = WorkspaceCodeIntelCache(normalized)
            self._workspace_caches[normalized] = cache
        return cache

    def _current_workspace_root(self) -> str:
        parent_agent = self.parent_agent
        if parent_agent is None:
            return self.workspace_root
        execution_context = getattr(parent_agent, "execution_context", None)
        worktree_path = getattr(execution_context, "worktree_path", None)
        if worktree_path:
            return os.path.abspath(str(worktree_path))
        workspace_root = getattr(execution_context, "workspace_root", None)
        if workspace_root:
            return os.path.abspath(str(workspace_root))
        return self.workspace_root

    def _current_allowed_roots(self, workspace_root: str) -> tuple[str, ...]:
        parent_agent = self.parent_agent
        if parent_agent is None:
            return self.allowed_roots
        execution_context = getattr(parent_agent, "execution_context", None)
        roots = tuple(getattr(execution_context, "allowed_roots", ()) or ())
        if roots:
            normalized = tuple(os.path.abspath(str(item)) for item in roots if item)
            return normalized or (workspace_root,)
        return self.allowed_roots or (workspace_root,)

    @staticmethod
    def _merge_query_metadata(
        result: CodeIntelQueryResult,
        **updates: Any,
    ) -> CodeIntelQueryResult:
        metadata = dict(result.metadata or {})
        metadata.update({key: value for key, value in updates.items() if value is not None})
        return CodeIntelQueryResult(
            status=result.status,
            provider_name=result.provider_name,
            workspace_root=result.workspace_root,
            items=list(result.items or []),
            fallback_tools=list(result.fallback_tools or []),
            error_message=result.error_message,
            metadata=metadata,
        )

    @staticmethod
    def _query_key(*parts: Any) -> str:
        return "|".join(str(part) for part in parts)

    @staticmethod
    def _file_fingerprint(path: str, *parts: Any) -> str:
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            mtime_ns = "missing"
        suffix = "|".join(str(part) for part in parts)
        return f"{os.path.abspath(path)}@{mtime_ns}|{suffix}"

    @staticmethod
    def _provider_state(provider: CodeIntelProvider) -> dict[str, Any]:
        export_fn = getattr(provider, "export_state", None)
        if callable(export_fn):
            try:
                payload = dict(export_fn() or {})
                if payload:
                    return payload
            except Exception:
                pass
        provider_name = getattr(provider, "provider_name", "codeintel")
        if provider_name == "lsp":
            return {
                "kind": "lsp",
                "providerName": provider_name,
                "serverCommand": list(getattr(provider, "server_command", []) or []),
                "serverEnv": dict(getattr(provider, "server_env", {}) or {}),
                "requestTimeoutMs": int(getattr(provider, "request_timeout_ms", 5000)),
                "diagnosticsWaitMs": int(getattr(provider, "diagnostics_wait_ms", 800)),
            }
        return {
            "kind": "builtin",
            "providerName": provider_name,
        }

    @staticmethod
    def _provider_from_state(state: dict[str, Any] | None) -> CodeIntelProvider:
        payload = dict(state or {})
        kind = str(payload.get("kind") or payload.get("providerName") or "lsp").strip().lower()
        if kind in {"lsp", "codeintel", "builtin"}:
            from .lsp import LSPCodeIntelProvider

            restore_fn = getattr(LSPCodeIntelProvider, "from_state", None)
            if callable(restore_fn):
                return restore_fn(payload)
            return LSPCodeIntelProvider(
                server_command=list(payload.get("serverCommand") or []),
                server_env=dict(payload.get("serverEnv") or {}),
                request_timeout_ms=int(payload.get("requestTimeoutMs") or 5000),
                diagnostics_wait_ms=int(payload.get("diagnosticsWaitMs") or 800),
            )
        from .lsp import LSPCodeIntelProvider

        return LSPCodeIntelProvider()

    def resolve_file_path(
        self,
        file_path: str,
        *,
        workspace_root: Optional[str] = None,
    ) -> tuple[str, str]:
        effective_workspace = os.path.abspath(workspace_root or self._current_workspace_root())
        allowed_roots = self._current_allowed_roots(effective_workspace)
        resolver = PathResolver(effective_workspace, allowed_roots=allowed_roots)
        resolved = resolver.resolve(
            file_path,
            cwd=effective_workspace,
            must_exist=True,
            expected_kind="file",
        )
        return effective_workspace, resolved

    def resolve_directory_path(
        self,
        directory_path: str,
        *,
        workspace_root: Optional[str] = None,
    ) -> tuple[str, str]:
        effective_workspace = os.path.abspath(workspace_root or self._current_workspace_root())
        allowed_roots = self._current_allowed_roots(effective_workspace)
        resolver = PathResolver(effective_workspace, allowed_roots=allowed_roots)
        resolved = resolver.resolve(
            directory_path,
            cwd=effective_workspace,
            must_exist=True,
            expected_kind="directory",
        )
        return effective_workspace, resolved

    def get_status(self, *, file_path: Optional[str] = None) -> CodeIntelAvailability:
        workspace_root = self._current_workspace_root()
        cache = self._cache_for(workspace_root)
        resolved_file = None
        if file_path:
            try:
                workspace_root, resolved_file = self.resolve_file_path(file_path, workspace_root=workspace_root)
                cache = self._cache_for(workspace_root)
            except PathResolutionError:
                resolved_file = None
        status = self.provider.get_status(workspace_root=workspace_root, file_path=resolved_file)
        cache.record_status(status)
        self.last_status = status
        metadata = dict(status.metadata or {})
        metadata["cache"] = cache.get_status_payload()
        return CodeIntelAvailability(
            available=status.available,
            provider_name=status.provider_name,
            workspace_root=status.workspace_root,
            server_command=list(status.server_command or []),
            reason=status.reason,
            metadata=metadata,
        )

    def _try_cached_query(
        self,
        *,
        workspace_root: str,
        action: str,
        query_key: str,
        fingerprint: str,
        provider_result: CodeIntelQueryResult,
    ) -> CodeIntelQueryResult:
        cached = self._cache_for(workspace_root).get_cached_query(
            action=action,
            query_key=query_key,
            fingerprint=fingerprint,
        )
        if cached is None:
            return self._merge_query_metadata(
                provider_result,
                cacheHit=False,
                offlineIndexAvailable=self._cache_for(workspace_root).get_status_payload()["offlineIndexAvailable"],
            )
        cached_result, cache_meta = cached
        return self._merge_query_metadata(
            cached_result,
            providerFallbackStatus=provider_result.status,
            providerFallbackError=provider_result.error_message,
            providerFallbackMetadata=dict(provider_result.metadata or {}),
            offlineFallbackUsed=True,
            cacheHit=True,
            cacheSource=action,
            cacheStale=cache_meta.get("stale"),
        )

    def find_definition(self, *, file_path: str, line: int, column: int) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        query_key = self._query_key(resolved_file, int(line), int(column))
        fingerprint = self._file_fingerprint(resolved_file, int(line), int(column))
        result = self.provider.find_definition(
            DefinitionQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
                line=int(line),
                column=int(column),
            )
        )
        if result.status == "ok":
            self._cache_for(workspace_root).record_query(
                action="definition",
                query_key=query_key,
                fingerprint=fingerprint,
                result=result,
            )
            return self._merge_query_metadata(result, cacheHit=False, cacheSource="provider")
        return self._try_cached_query(
            workspace_root=workspace_root,
            action="definition",
            query_key=query_key,
            fingerprint=fingerprint,
            provider_result=result,
        )

    def find_references(
        self,
        *,
        file_path: str,
        line: int,
        column: int,
        include_declaration: bool = False,
    ) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        query_key = self._query_key(resolved_file, int(line), int(column), bool(include_declaration))
        fingerprint = self._file_fingerprint(resolved_file, int(line), int(column), bool(include_declaration))
        result = self.provider.find_references(
            ReferenceQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
                line=int(line),
                column=int(column),
                include_declaration=bool(include_declaration),
            )
        )
        if result.status == "ok":
            self._cache_for(workspace_root).record_query(
                action="references",
                query_key=query_key,
                fingerprint=fingerprint,
                result=result,
            )
            return self._merge_query_metadata(result, cacheHit=False, cacheSource="provider")
        return self._try_cached_query(
            workspace_root=workspace_root,
            action="references",
            query_key=query_key,
            fingerprint=fingerprint,
            provider_result=result,
        )

    def get_document_symbols(self, *, file_path: str) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        result = self.provider.get_document_symbols(
            DocumentSymbolsQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
            )
        )
        if result.status == "ok":
            self._cache_for(workspace_root).record_document_symbols(resolved_file, result)
            return self._merge_query_metadata(result, cacheHit=False, cacheSource="provider")
        cached = self._cache_for(workspace_root).get_cached_document_symbols(resolved_file)
        if cached is None:
            return self._merge_query_metadata(result, cacheHit=False)
        cached_result, cache_meta = cached
        return self._merge_query_metadata(
            cached_result,
            providerFallbackStatus=result.status,
            providerFallbackError=result.error_message,
            providerFallbackMetadata=dict(result.metadata or {}),
            offlineFallbackUsed=True,
            cacheStale=cache_meta.get("stale"),
        )

    def get_workspace_symbols(self, *, query: str, limit: int = 50) -> CodeIntelQueryResult:
        workspace_root = self._current_workspace_root()
        cache = self._cache_for(workspace_root)
        result = self.provider.get_workspace_symbols(
            WorkspaceSymbolsQuery(
                workspace_root=workspace_root,
                query=str(query or ""),
                limit=max(1, int(limit)),
            )
        )
        if result.status == "ok" and list(result.items or []):
            return self._merge_query_metadata(
                result,
                cacheHit=False,
                cacheSource="provider",
                offlineIndexAvailable=cache.get_status_payload()["offlineIndexAvailable"],
            )
        cached = cache.search_symbols(
            query=str(query or ""),
            limit=max(1, int(limit)),
            provider_name=result.provider_name,
        )
        if cached is None:
            return self._merge_query_metadata(
                result,
                cacheHit=False,
                offlineIndexAvailable=cache.get_status_payload()["offlineIndexAvailable"],
            )
        return self._merge_query_metadata(
            cached,
            providerFallbackStatus=result.status if result.status != "ok" else "ok_empty",
            providerFallbackError=result.error_message,
            providerFallbackMetadata=dict(result.metadata or {}),
            offlineFallbackUsed=True,
            cacheHit=True,
            cacheSource="offline_index",
        )

    def get_diagnostics(self, *, file_path: str) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        result = self.provider.get_diagnostics(
            DiagnosticsQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
            )
        )
        if result.status == "ok":
            self._cache_for(workspace_root).record_diagnostics(resolved_file, result)
            return self._merge_query_metadata(result, cacheHit=False, cacheSource="provider")
        cached = self._cache_for(workspace_root).get_cached_diagnostics(resolved_file)
        if cached is None:
            return self._merge_query_metadata(result, cacheHit=False)
        cached_result, cache_meta = cached
        return self._merge_query_metadata(
            cached_result,
            providerFallbackStatus=result.status,
            providerFallbackError=result.error_message,
            providerFallbackMetadata=dict(result.metadata or {}),
            offlineFallbackUsed=True,
            cacheStale=cache_meta.get("stale"),
        )

    def prewarm_workspace(
        self,
        *,
        path_prefix: Optional[str] = None,
        max_files: int = 200,
        include_diagnostics: bool = True,
        force: bool = False,
        suffixes: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        workspace_root = self._current_workspace_root()
        cache = self._cache_for(workspace_root)
        scan_root = workspace_root
        if path_prefix:
            workspace_root, scan_root = self.resolve_directory_path(path_prefix, workspace_root=workspace_root)
            cache = self._cache_for(workspace_root)

        status = self.provider.get_status(workspace_root=workspace_root, file_path=None)
        cache.record_status(status)
        self.last_status = status
        normalized_suffixes = tuple(sorted({str(item).lower() for item in (suffixes or DEFAULT_CODEINTEL_SUFFIXES) if item}))
        summary = {
            "workspaceRoot": workspace_root,
            "scanRoot": scan_root,
            "providerAvailable": bool(status.available),
            "providerName": status.provider_name,
            "maxFiles": max(1, int(max_files)),
            "includeDiagnostics": bool(include_diagnostics),
            "force": bool(force),
            "suffixes": list(normalized_suffixes),
            "scannedFiles": 0,
            "indexedFiles": 0,
            "diagnosticsFiles": 0,
            "skippedFiles": 0,
            "errorCount": 0,
            "errors": [],
        }
        if not status.available:
            summary["reason"] = status.reason
            summary["offlineIndexAvailable"] = cache.get_status_payload()["offlineIndexAvailable"]
            cache.mark_prewarm(summary)
            return summary

        visited = 0
        for root, dirs, files in os.walk(scan_root):
            dirs[:] = [item for item in dirs if item not in DEFAULT_SKIP_DIRS]
            for file_name in files:
                if visited >= max(1, int(max_files)):
                    break
                if normalized_suffixes and os.path.splitext(file_name)[1].lower() not in normalized_suffixes:
                    continue
                file_path = os.path.abspath(os.path.join(root, file_name))
                entry = cache.file_entries.get(file_path)
                current_mtime_ns = None
                try:
                    current_mtime_ns = os.stat(file_path).st_mtime_ns
                except OSError as exc:
                    summary["errorCount"] += 1
                    summary["errors"].append({"path": file_path, "stage": "stat", "error": str(exc)})
                    continue
                already_fresh = (
                    not force
                    and entry is not None
                    and entry.mtime_ns == current_mtime_ns
                    and entry.document_symbols_payload is not None
                    and (not include_diagnostics or entry.diagnostics_payload is not None)
                )
                summary["scannedFiles"] += 1
                if already_fresh:
                    summary["skippedFiles"] += 1
                    visited += 1
                    continue

                symbols_result = self.provider.get_document_symbols(
                    DocumentSymbolsQuery(workspace_root=workspace_root, file_path=file_path)
                )
                if symbols_result.status == "ok":
                    cache.record_document_symbols(file_path, symbols_result)
                    summary["indexedFiles"] += 1
                else:
                    summary["errorCount"] += 1
                    summary["errors"].append(
                        {
                            "path": file_path,
                            "stage": "document_symbols",
                            "status": symbols_result.status,
                            "error": symbols_result.error_message,
                        }
                    )

                if include_diagnostics:
                    diagnostics_result = self.provider.get_diagnostics(
                        DiagnosticsQuery(workspace_root=workspace_root, file_path=file_path)
                    )
                    if diagnostics_result.status == "ok":
                        cache.record_diagnostics(file_path, diagnostics_result)
                        summary["diagnosticsFiles"] += 1
                    else:
                        summary["errorCount"] += 1
                        summary["errors"].append(
                            {
                                "path": file_path,
                                "stage": "diagnostics",
                                "status": diagnostics_result.status,
                                "error": diagnostics_result.error_message,
                            }
                        )
                visited += 1
            if visited >= max(1, int(max_files)):
                break

        summary["offlineIndexAvailable"] = cache.get_status_payload()["offlineIndexAvailable"]
        cache.mark_prewarm(summary)
        return summary

    def get_cache_status(self) -> dict[str, Any]:
        workspace_root = self._current_workspace_root()
        cache = self._cache_for(workspace_root)
        payload = cache.get_status_payload()
        payload["allowedRoots"] = list(self._current_allowed_roots(workspace_root))
        payload["workspaceCount"] = len(self._workspace_caches)
        return payload

    def export_state(self) -> dict[str, Any]:
        return {
            "workspaceRoot": self.workspace_root,
            "allowedRoots": list(self.allowed_roots),
            "provider": self._provider_state(self.provider),
            "lastStatus": self.last_status.to_dict() if self.last_status is not None else None,
            "workspaces": [cache.to_dict() for cache in self._workspace_caches.values()],
        }

    def restore_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(state or {})
        self.workspace_root = os.path.abspath(payload.get("workspaceRoot") or self.workspace_root)
        restored_allowed_roots = tuple(
            os.path.abspath(str(item))
            for item in list(payload.get("allowedRoots") or self.allowed_roots)
            if item
        )
        if restored_allowed_roots:
            self.allowed_roots = restored_allowed_roots
        last_status_payload = payload.get("lastStatus")
        self.last_status = CodeIntelAvailability.from_dict(last_status_payload) if last_status_payload else None
        restored_caches = [
            WorkspaceCodeIntelCache.from_dict(item)
            for item in list(payload.get("workspaces") or [])
            if item
        ]
        self._workspace_caches = {
            os.path.abspath(item.workspace_root): item
            for item in restored_caches
        }
        if self.workspace_root not in self._workspace_caches:
            self._workspace_caches[self.workspace_root] = WorkspaceCodeIntelCache(self.workspace_root)
        restored_workspace_roots = sorted(self._workspace_caches.keys())
        return {
            "status": "restored",
            "restoredItems": ["codeintel_cache"],
            "metadata": {
                "workspaceCount": len(restored_workspace_roots),
                "workspaceRoots": restored_workspace_roots,
            },
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any] | None,
        *,
        parent_agent: Any | None = None,
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        provider: Optional[CodeIntelProvider] = None,
    ) -> "CodeIntelManager":
        payload = dict(state or {})
        manager = cls(
            provider=provider or cls._provider_from_state(payload.get("provider") or {}),
            parent_agent=parent_agent,
            workspace_root=workspace_root or payload.get("workspaceRoot"),
            allowed_roots=allowed_roots or payload.get("allowedRoots"),
        )
        manager.restore_state(payload)
        return manager

    def close(self) -> None:
        self.provider.close()


__all__ = ["CodeIntelManager", "DEFAULT_CODEINTEL_SUFFIXES"]
