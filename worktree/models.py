"""Worktree values exchanged by module implementations and tools."""
from __future__ import annotations
from dataclasses import dataclass
import os
import time
from typing import Any


@dataclass(slots=True)
class GitWorktreeInfo:
    path: str
    branch: str | None
    head: str | None
    bare: bool = False
    detached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "branch": self.branch, "head": self.head,
                "bare": self.bare, "detached": self.detached}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GitWorktreeInfo":
        data = dict(payload or {})
        return cls(os.path.abspath(str(data.get("path") or "")), data.get("branch"),
                   data.get("head"), bool(data.get("bare", False)), bool(data.get("detached", False)))


@dataclass(slots=True)
class GitWorktreeSession:
    original_cwd: str
    worktree: GitWorktreeInfo
    base_head: str | None
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {"originalCwd": self.original_cwd, "worktree": self.worktree.to_dict(),
                "baseHead": self.base_head, "createdAt": self.created_at}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GitWorktreeSession | None":
        data = dict(payload or {})
        if not data.get("worktree"):
            return None
        return cls(os.path.abspath(str(data.get("originalCwd") or os.getcwd())),
                   GitWorktreeInfo.from_dict(data["worktree"]), data.get("baseHead"),
                   float(data.get("createdAt") or time.time()))
