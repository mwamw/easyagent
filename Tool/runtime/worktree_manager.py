"""Git worktree helpers for future Claude style isolated execution."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class GitWorktreeInfo:
    path: str
    branch: Optional[str]
    head: Optional[str]
    bare: bool = False
    detached: bool = False


class WorktreeManager:
    def __init__(self, repo_root: str, *, git_binary: str = "git", storage_dir: Optional[str] = None):
        self.repo_root = os.path.abspath(repo_root)
        self.git_binary = git_binary
        self.storage_dir = os.path.abspath(
            storage_dir or os.path.join(os.path.dirname(self.repo_root), ".easyagent-worktrees")
        )
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
        subprocess.run(
            [self.git_binary, "worktree", "add", "-b", branch_name, worktree_path, base_ref],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return GitWorktreeInfo(path=worktree_path, branch=branch_name, head=None)

    def remove_worktree(self, path: str, *, force: bool = False) -> None:
        args = [self.git_binary, "worktree", "remove"]
        if force:
            args.append("--force")
        args.append(os.path.abspath(path))
        subprocess.run(
            args,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

    def list_worktrees(self) -> list[GitWorktreeInfo]:
        completed = subprocess.run(
            [self.git_binary, "worktree", "list", "--porcelain"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
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

    def prune(self) -> None:
        subprocess.run(
            [self.git_binary, "worktree", "prune"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
