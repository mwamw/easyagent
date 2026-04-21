"""Runtime manager for workspace-aware code intelligence access."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from Tool.runtime import PathResolver, PathResolutionError

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


class CodeIntelManager:
    def __init__(
        self,
        *,
        provider: CodeIntelProvider,
        parent_agent: Any | None = None,
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
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

    def get_status(self, *, file_path: Optional[str] = None) -> CodeIntelAvailability:
        workspace_root = self._current_workspace_root()
        resolved_file = None
        if file_path:
            try:
                workspace_root, resolved_file = self.resolve_file_path(file_path, workspace_root=workspace_root)
            except PathResolutionError:
                resolved_file = None
        status = self.provider.get_status(workspace_root=workspace_root, file_path=resolved_file)
        self.last_status = status
        return status

    def find_definition(self, *, file_path: str, line: int, column: int) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        return self.provider.find_definition(
            DefinitionQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
                line=int(line),
                column=int(column),
            )
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
        return self.provider.find_references(
            ReferenceQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
                line=int(line),
                column=int(column),
                include_declaration=bool(include_declaration),
            )
        )

    def get_document_symbols(self, *, file_path: str) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        return self.provider.get_document_symbols(
            DocumentSymbolsQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
            )
        )

    def get_workspace_symbols(self, *, query: str, limit: int = 50) -> CodeIntelQueryResult:
        workspace_root = self._current_workspace_root()
        return self.provider.get_workspace_symbols(
            WorkspaceSymbolsQuery(
                workspace_root=workspace_root,
                query=str(query or ""),
                limit=max(1, int(limit)),
            )
        )

    def get_diagnostics(self, *, file_path: str) -> CodeIntelQueryResult:
        workspace_root, resolved_file = self.resolve_file_path(file_path)
        return self.provider.get_diagnostics(
            DiagnosticsQuery(
                workspace_root=workspace_root,
                file_path=resolved_file,
            )
        )

    def close(self) -> None:
        self.provider.close()


__all__ = ["CodeIntelManager"]
