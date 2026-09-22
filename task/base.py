"""Public task-management contract, independent of storage and scheduling."""
from abc import ABC, abstractmethod
from typing import Any
from .models import TaskRecord, TaskStatus


class BaseTaskService(ABC):
    def export_state(self) -> None:
        """Record capability presence without copying the external task store."""
        return None

    def restore_state(self, state: dict[str, Any] | None) -> None:
        if state is not None:
            raise ValueError("This task implementation does not restore session-local state")

    @abstractmethod
    def create_task(self, *, title: str, description: str = "",
                    status: TaskStatus = TaskStatus.OPEN, owner: str | None = None,
                    parent_task_id: str | None = None, metadata: dict[str, Any] | None = None,
                    task_id: str | None = None) -> TaskRecord:
        raise NotImplementedError

    @abstractmethod
    def get_task(self, task_id: str) -> TaskRecord:
        raise NotImplementedError

    @abstractmethod
    def update_task(self, task_id: str, *, title: str | None = None,
                    description: str | None = None, status: TaskStatus | None = None,
                    owner: str | None = None, parent_task_id: str | None = None,
                    metadata: dict[str, Any] | None = None, merge_metadata: bool = True) -> TaskRecord:
        raise NotImplementedError

    @abstractmethod
    def list_tasks(self, *, status: TaskStatus | None = None, owner: str | None = None,
                   parent_task_id: str | None = None, metadata_filters: dict[str, Any] | None = None,
                   limit: int = 100) -> list[TaskRecord]:
        raise NotImplementedError
