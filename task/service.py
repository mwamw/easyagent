"""Task service."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from core.Exception import TaskNotFoundError

from .models import TaskRecord, TaskStatus
from .store import BaseTaskStore


class TaskService:
    def __init__(self, store: BaseTaskStore):
        self.store = store

    def create_task(
        self,
        *,
        title: str,
        description: str = "",
        status: TaskStatus = TaskStatus.OPEN,
        owner: str | None = None,
        parent_task_id: str | None = None,
        metadata: dict | None = None,
        task_id: str | None = None,
    ) -> TaskRecord:
        now = datetime.now()
        record = TaskRecord(
            task_id=task_id or f"task_{uuid4().hex[:12]}",
            title=title.strip(),
            description=description,
            status=status,
            owner=owner,
            parent_task_id=parent_task_id,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        return self.store.create_task(record)

    def get_task(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if task is None:
            raise TaskNotFoundError(f"任务不存在: {task_id}")
        return task

    def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: TaskStatus | None = None,
        owner: str | None = None,
        parent_task_id: str | None = None,
        metadata: dict | None = None,
        merge_metadata: bool = True,
    ) -> TaskRecord:
        current = self.get_task(task_id)
        next_metadata = dict(current.metadata)
        if metadata is not None:
            next_metadata = {**next_metadata, **metadata} if merge_metadata else dict(metadata)
        updated = current.model_copy(
            update={
                "title": title if title is not None else current.title,
                "description": description if description is not None else current.description,
                "status": status if status is not None else current.status,
                "owner": owner if owner is not None else current.owner,
                "parent_task_id": parent_task_id if parent_task_id is not None else current.parent_task_id,
                "metadata": next_metadata,
                "updated_at": datetime.now(),
            }
        )
        return self.store.update_task(updated)

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        owner: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
    ) -> list[TaskRecord]:
        return self.store.list_tasks(
            status=status,
            owner=owner,
            parent_task_id=parent_task_id,
            limit=limit,
        )

