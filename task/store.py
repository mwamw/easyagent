"""Task storage backends."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from .models import TaskRecord, TaskStatus


DEFAULT_TASK_DB_PATH = "db/easyagent_tasks.db"


class BaseTaskStore(ABC):
    @abstractmethod
    def create_task(self, task: TaskRecord) -> TaskRecord:
        raise NotImplementedError

    @abstractmethod
    def get_task(self, task_id: str) -> TaskRecord | None:
        raise NotImplementedError

    @abstractmethod
    def update_task(self, task: TaskRecord) -> TaskRecord:
        raise NotImplementedError

    @abstractmethod
    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        owner: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        raise NotImplementedError


class InMemoryTaskStore(BaseTaskStore):
    def __init__(self):
        self._lock = RLock()
        self._tasks: dict[str, TaskRecord] = {}

    def create_task(self, task: TaskRecord) -> TaskRecord:
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task: TaskRecord) -> TaskRecord:
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        owner: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        with self._lock:
            tasks = list(self._tasks.values())
        if status is not None:
            tasks = [task for task in tasks if task.status == status]
        if owner is not None:
            tasks = [task for task in tasks if task.owner == owner]
        if parent_task_id is not None:
            tasks = [task for task in tasks if task.parent_task_id == parent_task_id]
        tasks.sort(key=lambda item: item.updated_at, reverse=True)
        return tasks[:limit]


class SQLiteTaskStore(BaseTaskStore):
    def __init__(self, db_path: str = DEFAULT_TASK_DB_PATH):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    owner TEXT,
                    parent_task_id TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_task(self, task: TaskRecord) -> TaskRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    task_id, title, description, status, owner, parent_task_id,
                    metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.title,
                    task.description,
                    task.status.value,
                    task.owner,
                    task.parent_task_id,
                    json.dumps(task.metadata, ensure_ascii=False, default=str),
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                ),
            )
        return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def update_task(self, task: TaskRecord) -> TaskRecord:
        return self.create_task(task)

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        owner: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        query = "SELECT * FROM tasks"
        conditions: list[str] = []
        params: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        if owner is not None:
            conditions.append("owner = ?")
            params.append(owner)
        if parent_task_id is not None:
            conditions.append("parent_task_id = ?")
            params.append(parent_task_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            title=row["title"],
            description=row["description"],
            status=TaskStatus(row["status"]),
            owner=row["owner"],
            parent_task_id=row["parent_task_id"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

