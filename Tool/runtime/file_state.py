"""In-memory file version tracking for safe local file mutations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from typing import Optional


@dataclass(frozen=True, slots=True)
class FileVersionSnapshot:
    path: str
    mtime_ns: int
    size: int
    inode: Optional[int] = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "path": self.path,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "inode": self.inode,
        }


file_read_timestamps: dict[str, FileVersionSnapshot] = {}
_LOCK = RLock()


def snapshot_file_version(path: str) -> FileVersionSnapshot:
    normalized = os.path.abspath(path)
    stats = os.stat(normalized)
    return FileVersionSnapshot(
        path=normalized,
        mtime_ns=stats.st_mtime_ns,
        size=stats.st_size,
        inode=getattr(stats, "st_ino", None),
    )


def get_recorded_file_version(path: str) -> FileVersionSnapshot | None:
    normalized = os.path.abspath(path)
    with _LOCK:
        return file_read_timestamps.get(normalized)


def remember_file_version(path: str) -> FileVersionSnapshot:
    snapshot = snapshot_file_version(path)
    with _LOCK:
        file_read_timestamps[snapshot.path] = snapshot
    return snapshot


def clear_file_read_timestamps() -> None:
    with _LOCK:
        file_read_timestamps.clear()


def recorded_file_is_current(path: str) -> tuple[bool, FileVersionSnapshot | None, FileVersionSnapshot]:
    current = snapshot_file_version(path)
    recorded = get_recorded_file_version(current.path)
    return recorded == current, recorded, current


__all__ = [
    "FileVersionSnapshot",
    "file_read_timestamps",
    "snapshot_file_version",
    "get_recorded_file_version",
    "remember_file_version",
    "clear_file_read_timestamps",
    "recorded_file_is_current",
]
