"""Worktree lifecycle contract; Git invocation and storage belong to implementations."""
from abc import ABC, abstractmethod
from typing import Any
from .models import GitWorktreeInfo, GitWorktreeSession


class BaseWorktreeManager(ABC):
    @abstractmethod
    def create_worktree(self, name: str, *, base_ref: str = "HEAD",
                        branch_prefix: str = "easyagent/") -> GitWorktreeInfo:
        raise NotImplementedError

    @abstractmethod
    def remove_worktree(self, path: str, *, force: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_active_session(self) -> GitWorktreeSession | None:
        raise NotImplementedError

    @abstractmethod
    def enter_worktree(self, name: str | None = None, *, base_ref: str = "HEAD",
                       branch_prefix: str = "easyagent/") -> GitWorktreeSession:
        raise NotImplementedError

    @abstractmethod
    def exit_worktree(self, action: str, *, discard_changes: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self, *, action: str = "keep", discard_changes: bool = False) -> dict[str, Any] | None:
        raise NotImplementedError
