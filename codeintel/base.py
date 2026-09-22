"""Agent-facing code intelligence contract; providers and caching are implementation details."""
from abc import ABC, abstractmethod
from typing import Any
from runtime.agent import AgentRuntime
from .models import CodeIntelAvailability, CodeIntelQueryResult


class BaseCodeIntel(ABC):
    @abstractmethod
    def bind(self, runtime: AgentRuntime) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_status(self, *, file_path: str | None = None) -> CodeIntelAvailability:
        raise NotImplementedError

    @abstractmethod
    def find_definition(self, *, file_path: str, line: int, column: int) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def find_references(self, *, file_path: str, line: int, column: int,
                        include_declaration: bool = False) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def get_document_symbols(self, *, file_path: str) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def get_workspace_symbols(self, *, query: str, limit: int = 50) -> CodeIntelQueryResult:
        raise NotImplementedError

    @abstractmethod
    def get_diagnostics(self, *, file_path: str) -> CodeIntelQueryResult:
        raise NotImplementedError

    def get_cache_status(self) -> dict[str, Any]:
        return {"available": False, "reason": "This implementation does not expose a cache"}

    def prewarm_workspace(self, *, path_prefix: str | None = None, max_files: int = 200,
                          include_diagnostics: bool = True, force: bool = False) -> dict[str, Any]:
        return {"available": False, "reason": "This implementation does not support prewarming"}

    def close(self) -> None:
        """Release implementation-owned resources."""
