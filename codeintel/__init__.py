"""Public exports for EasyAgent code intelligence."""

from .cache import CachedFileEntry, CachedQueryEntry, WorkspaceCodeIntelCache
from .base import BaseCodeIntel
from .manager import CodeIntelManager
from .models import (
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
from .provider import CodeIntelProvider, DEFAULT_FALLBACK_TOOLS
from .lsp import LSPClient, LSPClientError, LSPCodeIntelProvider, path_to_uri, uri_to_path

__all__ = [
    "BaseCodeIntel",
    "CachedFileEntry",
    "CachedQueryEntry",
    "CodeIntelAvailability",
    "CodeIntelManager",
    "CodeIntelProvider",
    "CodeIntelQueryResult",
    "CodeLocation",
    "CodePosition",
    "CodeRange",
    "DEFAULT_FALLBACK_TOOLS",
    "DefinitionQuery",
    "DiagnosticRecord",
    "DiagnosticsQuery",
    "DocumentSymbolsQuery",
    "LSPClient",
    "LSPClientError",
    "LSPCodeIntelProvider",
    "ReferenceQuery",
    "SymbolRecord",
    "WorkspaceCodeIntelCache",
    "WorkspaceSymbolsQuery",
    "path_to_uri",
    "uri_to_path",
]
