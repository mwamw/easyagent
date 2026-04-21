"""Abstract interfaces for code intelligence providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import (
    CodeIntelAvailability,
    CodeIntelQueryResult,
    DefinitionQuery,
    DiagnosticsQuery,
    DocumentSymbolsQuery,
    ReferenceQuery,
    WorkspaceSymbolsQuery,
)


DEFAULT_FALLBACK_TOOLS = ("FileRead", "Grep", "Glob")


class CodeIntelProvider(ABC):
    provider_name: str = "codeintel"

    @abstractmethod
    def get_status(
        self,
        *,
        workspace_root: str,
        file_path: Optional[str] = None,
    ) -> CodeIntelAvailability:
        raise NotImplementedError

    @abstractmethod
    def find_definition(self, query: DefinitionQuery) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def find_references(self, query: ReferenceQuery) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def get_document_symbols(self, query: DocumentSymbolsQuery) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def get_workspace_symbols(self, query: WorkspaceSymbolsQuery) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def get_diagnostics(self, query: DiagnosticsQuery) -> CodeIntelQueryResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


__all__ = ["CodeIntelProvider", "DEFAULT_FALLBACK_TOOLS"]
