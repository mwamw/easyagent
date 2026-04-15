"""Git worktree helpers with session state for Claude-style isolated execution."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
import time
from typing import Any, Optional
from uuid import uuid4


@dataclass(slots=True)
class GitWorktreeInfo:
    path: str
    branch: Optional[str]
    head: Optional[str]
    bare: bool = False
    detached: bool = False


@dataclass(slots=True)
class GitWorktreeSession:
    original_cwd: str
    worktree: GitWorktreeInfo
    base_head: Optional[str]
    created_at: float


class WorktreeManager:
    def __init__(
        self,
        repo_root: str,
        *,
        git_binary: str = "git",
        storage_dir: Optional[str] = None,
        original_cwd: Optional[str] = None,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.git_binary = git_binary
        self.storage_dir = os.path.abspath(
            storage_dir or os.path.join(os.path.dirname(self.repo_root), ".easyagent-worktrees")
        )
        self.original_cwd = os.path.abspath(original_cwd or self.repo_root)
        self._active_session: Optional[GitWorktreeSession] = None
        os.makedirs(self.storage_dir, exist_ok=True)

    @classmethod
    def detect_repo_root(cls, start_path: str, *, git_binary: str = "git") -> str:
        completed = subprocess.run(
            [git_binary, "rev-parse", "--show-toplevel"],
            cwd=os.path.abspath(start_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    @staticmethod
    def sanitize_name(name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name).strip("._")
        return cleaned or "worktree"

    @staticmethod
    def generate_name(prefix: str = "worktree") -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return WorktreeManager.sanitize_name(f"{prefix}-{timestamp}-{uuid4().hex[:6]}")

    def _run_git(
        self,
        args: list[str],
        *,
        cwd: Optional[str] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.git_binary, *args],
            cwd=os.path.abspath(cwd or self.repo_root),
            capture_output=True,
            text=True,
            check=check,
        )

    def _resolve_head(self, cwd: str) -> Optional[str]:
        completed = self._run_git(["rev-parse", "HEAD"], cwd=cwd, check=False)
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    def _count_dirty_files(self, cwd: str) -> int:
        completed = self._run_git(["status", "--porcelain"], cwd=cwd)
        return len([line for line in completed.stdout.splitlines() if line.strip()])

    def _count_ahead_commits(self, cwd: str, base_head: Optional[str]) -> int:
        if not base_head:
            return 0
        completed = self._run_git(["rev-list", "--count", f"{base_head}..HEAD"], cwd=cwd, check=False)
        if completed.returncode != 0:
            return 0
        try:
            return int(completed.stdout.strip() or "0")
        except ValueError:
            return 0

    def create_worktree(
        self,
        name: str,
        *,
        base_ref: str = "HEAD",
        branch_prefix: str = "easyagent/",
    ) -> GitWorktreeInfo:
        safe_name = self.sanitize_name(name)
        branch_name = f"{branch_prefix}{safe_name}"
        worktree_path = os.path.join(self.storage_dir, safe_name)
        self._run_git(["worktree", "add", "-b", branch_name, worktree_path, base_ref], cwd=self.repo_root)
        return GitWorktreeInfo(
            path=worktree_path,
            branch=branch_name,
            head=self._resolve_head(worktree_path),
        )

    def remove_worktree(self, path: str, *, force: bool = False) -> None:
        args = [self.git_binary, "worktree", "remove"]
        if force:
            args.append("--force")
        args.append(os.path.abspath(path))
        subprocess.run(args, cwd=self.repo_root, capture_output=True, text=True, check=True)

    def list_worktrees(self) -> list[GitWorktreeInfo]:
        completed = self._run_git(["worktree", "list", "--porcelain"], cwd=self.repo_root)
        blocks = [block.strip() for block in completed.stdout.split("\n\n") if block.strip()]
        result: list[GitWorktreeInfo] = []
        for block in blocks:
            path = None
            branch = None
            head = None
            bare = False
            detached = False
            for line in block.splitlines():
                if line.startswith("worktree "):
                    path = line.split(" ", 1)[1]
                elif line.startswith("branch "):
                    branch = line.split(" ", 1)[1]
                elif line.startswith("HEAD "):
                    head = line.split(" ", 1)[1]
                elif line == "bare":
                    bare = True
                elif line == "detached":
                    detached = True
            if path:
                result.append(
                    GitWorktreeInfo(
                        path=path,
                        branch=branch,
                        head=head,
                        bare=bare,
                        detached=detached,
                    )
                )
        return result

    def get_active_session(self) -> Optional[GitWorktreeSession]:
        return self._active_session

    def get_active_worktree(self) -> Optional[GitWorktreeInfo]:
        if self._active_session is None:
            return None
        return self._active_session.worktree

    def enter_worktree(
        self,
        name: Optional[str] = None,
        *,
        base_ref: str = "HEAD",
        branch_prefix: str = "easyagent/",
    ) -> GitWorktreeSession:
        if self._active_session is not None:
            raise RuntimeError(f"当前已有活动 worktree: {self._active_session.worktree.path}")

        worktree_name = name or self.generate_name("agent")
        worktree = self.create_worktree(worktree_name, base_ref=base_ref, branch_prefix=branch_prefix)
        session = GitWorktreeSession(
            original_cwd=self.original_cwd,
            worktree=worktree,
            base_head=worktree.head,
            created_at=time.time(),
        )
        self._active_session = session
        return session

    def exit_worktree(self, action: str, *, discard_changes: bool = False) -> dict[str, Any]:
        if self._active_session is None:
            raise RuntimeError("当前没有活动 worktree。")
        if action not in {"keep", "remove"}:
            raise ValueError(f"不支持的 exit action: {action}")
        if action == "keep" and discard_changes:
            raise ValueError("action=keep 时不能设置 discard_changes=true。")

        session = self._active_session
        discarded_files = 0
        discarded_commits = 0

        if action == "remove":
            if discard_changes:
                discarded_files = self._count_dirty_files(session.worktree.path)
                discarded_commits = self._count_ahead_commits(session.worktree.path, session.base_head)
            self.remove_worktree(session.worktree.path, force=discard_changes)
            self.prune()

        self._active_session = None
        return {
            "action": action,
            "originalCwd": session.original_cwd,
            "worktreePath": session.worktree.path,
            "worktreeBranch": session.worktree.branch,
            "discardedFiles": discarded_files,
            "discardedCommits": discarded_commits,
            "message": (
                f"已退出 worktree: {session.worktree.path}"
                if action == "keep"
                else f"已移除 worktree: {session.worktree.path}"
            ),
        }

    def prune(self) -> None:
        self._run_git(["worktree", "prune"], cwd=self.repo_root)


__all__ = ["GitWorktreeInfo", "GitWorktreeSession", "WorktreeManager"]
