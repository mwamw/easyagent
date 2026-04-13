"""Filesystem safety helpers used by local coding tools."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from .path_resolver import PathResolutionError, PathResolver


class FilesystemAccessError(PermissionError):
    """Raised when a filesystem operation violates EasyAgent safety policy."""


class FilesystemGuard:
    def __init__(self, workspace_root: str, allowed_roots: Optional[Iterable[str]] = None):
        self.resolver = PathResolver(workspace_root=workspace_root, allowed_roots=allowed_roots)

    @property
    def workspace_root(self) -> str:
        return self.resolver.workspace_root

    @property
    def allowed_roots(self) -> tuple[str, ...]:
        return self.resolver.allowed_roots

    def resolve_read_path(self, path: str, *, cwd: Optional[str] = None) -> str:
        return self.resolver.resolve(path, cwd=cwd, must_exist=True, expected_kind="file")

    def resolve_directory(self, path: str, *, cwd: Optional[str] = None, must_exist: bool = True) -> str:
        return self.resolver.resolve(path, cwd=cwd, must_exist=must_exist, expected_kind="dir")

    def resolve_write_path(
        self,
        path: str,
        *,
        cwd: Optional[str] = None,
        create_parents: bool = False,
    ) -> str:
        resolved = self.resolver.resolve(path, cwd=cwd, must_exist=False)
        parent = os.path.dirname(resolved) or self.workspace_root
        self.resolver.ensure_allowed(parent)

        if create_parents:
            os.makedirs(parent, exist_ok=True)
        elif not os.path.isdir(parent):
            raise FilesystemAccessError(f"父目录不存在: {parent}")
        return resolved

    def ensure_parent_writable(self, path: str) -> None:
        parent = os.path.dirname(path) or self.workspace_root
        if not os.path.isdir(parent):
            raise FilesystemAccessError(f"父目录不存在: {parent}")
        if not os.access(parent, os.W_OK):
            raise FilesystemAccessError(f"父目录不可写: {parent}")

    def ensure_file_readable(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FilesystemAccessError(f"文件不存在: {path}")
        if not os.access(path, os.R_OK):
            raise FilesystemAccessError(f"文件不可读: {path}")

    def ensure_file_writable(self, path: str) -> None:
        if os.path.exists(path) and not os.access(path, os.W_OK):
            raise FilesystemAccessError(f"文件不可写: {path}")
        self.ensure_parent_writable(path)

    def validate_glob_root(self, path: Optional[str], *, cwd: Optional[str] = None) -> str:
        target = path or self.workspace_root
        return self.resolve_directory(target, cwd=cwd, must_exist=True)

    def safe_join(self, *parts: str) -> str:
        return self.resolver.resolve(os.path.join(*parts), must_exist=False)
