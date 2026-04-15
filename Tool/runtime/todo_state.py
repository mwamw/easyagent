"""In-memory todo state tracking for Claude-style TodoWrite."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TodoItemSnapshot:
    content: str
    status: str
    active_form: str

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content,
            "status": self.status,
            "activeForm": self.active_form,
        }


_LOCK = RLock()
todo_items: list[TodoItemSnapshot] = []


def normalize_todo_item(item: dict[str, str] | TodoItemSnapshot) -> TodoItemSnapshot:
    if isinstance(item, TodoItemSnapshot):
        return item
    return TodoItemSnapshot(
        content=str(item.get("content", "")).strip(),
        status=str(item.get("status", "")).strip(),
        active_form=str(item.get("activeForm", "")).strip(),
    )


def get_todo_items() -> list[TodoItemSnapshot]:
    with _LOCK:
        return list(todo_items)


def set_todo_items(items: Iterable[dict[str, str] | TodoItemSnapshot]) -> tuple[list[TodoItemSnapshot], list[TodoItemSnapshot]]:
    normalized = [normalize_todo_item(item) for item in items]
    with _LOCK:
        previous = list(todo_items)
        todo_items[:] = normalized
        current = list(todo_items)
    return previous, current


def clear_todo_items() -> None:
    with _LOCK:
        todo_items.clear()


__all__ = [
    "TodoItemSnapshot",
    "todo_items",
    "normalize_todo_item",
    "get_todo_items",
    "set_todo_items",
    "clear_todo_items",
]
