"""Path resolution helpers for local coding tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


class PathResolutionError(ValueError):
    """Raised when a user supplied path cannot be resolved safely."""


class PathResolver:
    def __init__(self, workspace_root: str, allowed_roots: Optional[Iterable[str]] = None):
        if not workspace_root:
            raise PathResolutionError("workspace_root 不能为空。")
        self.workspace_root = self._normalize_existing_dir(workspace_root)
        roots = list(allowed_roots or [])
        normalized_roots = [self.workspace_root]
        for root in roots:
            normalized = self._normalize_existing_dir(root)
            if normalized not in normalized_roots:
                normalized_roots.append(normalized)
        self.allowed_roots = tuple(normalized_roots)

    @staticmethod
    def _normalize_existing_dir(value: str) -> str:
        normalized = str(Path(value).expanduser().resolve(strict=False))
        if not os.path.isdir(normalized):
            raise PathResolutionError(f"目录不存在: {value}")
        return normalized

    def resolve(
        self,
        path: str,
        *,
        cwd: Optional[str] = None,
        must_exist: bool = False,
        expected_kind: Optional[str] = None,
    ) -> str:
        if not path:
            raise PathResolutionError("路径不能为空。")

        base_dir = cwd or self.workspace_root
        base_path = Path(base_dir).expanduser().resolve(strict=False)
        if not base_path.is_dir():
            raise PathResolutionError(f"cwd 不是有效目录: {base_dir}")

        raw_path = Path(path).expanduser()
        candidate = raw_path if raw_path.is_absolute() else base_path / raw_path
        resolved = str(candidate.resolve(strict=False))
        self.ensure_allowed(resolved)

        if must_exist and not os.path.exists(resolved):
            raise PathResolutionError(f"路径不存在: {resolved}")

        if expected_kind == "file" and os.path.exists(resolved) and not os.path.isfile(resolved):
            raise PathResolutionError(f"期望文件路径，实际不是文件: {resolved}")
        if expected_kind == "dir" and os.path.exists(resolved) and not os.path.isdir(resolved):
            raise PathResolutionError(f"期望目录路径，实际不是目录: {resolved}")

        return resolved

    def ensure_allowed(self, path: str) -> None:
        normalized = str(Path(path).expanduser().resolve(strict=False))
        for root in self.allowed_roots:
            try:
                if os.path.commonpath([normalized, root]) == root:
                    return
            except ValueError:
                continue
        raise PathResolutionError(f"路径超出允许的根目录范围: {normalized}")

    def relative_to_workspace(self, path: str) -> str:
        normalized = self.resolve(path, must_exist=False)
        return os.path.relpath(normalized, self.workspace_root)
