"""Task domain models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskRecord(BaseModel):
    task_id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.OPEN
    owner: str | None = None
    parent_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

