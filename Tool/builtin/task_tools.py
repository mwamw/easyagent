"""Structured task tools for EasyAgent."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from core.Exception import TaskNotFoundError
from task import InMemoryTaskStore, TaskService, TaskStatus

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import dump_tool_payload, format_structured_display


TASK_CREATE_PROMPT = """Use this tool to create a structured task record. This helps track progress, organize complex work, and give the user visibility into the task.

When to use:
- Complex work with multiple steps or deliverables.
- Tasks that need an owner, parent task, or persistent status.
- Do not create tasks for every trivial operation.

Fields:
- title: a concise, actionable task title.
- description: requirements, context, constraints, and acceptance criteria.
- status: open by default; in_progress only when work has started.
- owner: optional agent or person responsible for the task.
- parent_task_id: an existing parent task ID.
- metadata: additional structured context.

Statuses are open, in_progress, blocked, completed, and cancelled.
Use TaskUpdate to record progress, TaskGet for full details, and TaskList to inspect child tasks.
Agent creates a child task automatically when the parent is task-bound; do not duplicate it.
TodoWrite, when available, is a lighter interface for maintaining a complete checklist."""


TASK_GET_PROMPT = """Use this tool to retrieve a task by its ID.

When to use:
- To get the full description and context before starting work.
- Before updating a task whose current status is unknown.
- After a subagent finishes, to inspect its associated task.

Returns task_id, title, description, status, owner, parent_task_id, metadata, created_at, and updated_at.
Use TaskList for multiple tasks or child tasks. Check for task_not_found instead of inventing an ID."""


TASK_UPDATE_PROMPT = """Use this tool to update an existing task.

When to use:
- Set in_progress when work begins.
- Set completed only when the task is fully accomplished.
- Set blocked when progress requires a dependency or external input.
- Set cancelled when the task is no longer needed.

Pass task_id and only the fields to change. Omitted fields are preserved.
metadata is merged into existing metadata.
Do not mark a task completed when tests fail or implementation is incomplete.
Do not rewrite unrelated task fields merely to update status."""


TASK_LIST_PROMPT = """Use this tool to list structured tasks and see overall progress.
- Filter by status, owner, or parent_task_id.
- Use parent_task_id to list children of a known task.
- limit defaults to 100 and cannot exceed 500.
- Use TaskGet when full details of a specific task are needed.
- Reuse returned task IDs when updating tasks."""



class TaskCreateParams(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(default="", description="Task description")
    status: TaskStatus = Field(default=TaskStatus.OPEN, description="Initial status")
    owner: Optional[str] = Field(default=None, description="Task owner")
    parent_task_id: Optional[str] = Field(default=None, description="Parent task ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskGetParams(BaseModel):
    task_id: str = Field(description="Task ID")


class TaskUpdateParams(BaseModel):
    task_id: str = Field(description="Task ID")
    title: Optional[str] = Field(default=None, description="New title")
    description: Optional[str] = Field(default=None, description="New description")
    status: Optional[TaskStatus] = Field(default=None, description="New status")
    owner: Optional[str] = Field(default=None, description="New owner")
    parent_task_id: Optional[str] = Field(default=None, description="Parent task ID")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Metadata fields to merge")


class TaskListParams(BaseModel):
    status: Optional[TaskStatus] = Field(default=None, description="Filter by status")
    owner: Optional[str] = Field(default=None, description="Filter by owner")
    parent_task_id: Optional[str] = Field(default=None, description="Filter by parent task ID")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum number of results")


def _service_or_default(service: TaskService | None) -> TaskService:
    return service or TaskService(InMemoryTaskStore())


class _TaskToolBase(Tool):
    def __init__(self, *, service: TaskService, **kwargs):
        self.service = service
        super().__init__(**kwargs)


class TaskCreateTool(_TaskToolBase):
    def __init__(self, *, service: TaskService):
        super().__init__(
            service=service,
            name="TaskCreate",
            description="Creates a persistent structured task with optional parent and metadata.",
            parameters=TaskCreateParams,
            guidance="Use clear titles and acceptance criteria. Prefer TodoWrite for a simple checklist when available.",
            prompt=TASK_CREATE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["task", "planning"],
            risk_categories=["side_effect"],
            side_effect_level="low",
            resource_scope=["task", "runtime"],
        )


    def run(self, parameters: dict) -> ToolResult:
        task = self.service.create_task(**parameters)
        payload = task.model_dump(mode="python")
        return ToolResult.success(
            content=f"已创建任务 {task.task_id}",
            display_text=format_structured_display(
                f"已创建任务 {task.task_id}",
                payload,
            ),
            structured_data=payload,
            metadata={"task_id": task.task_id},
        )


class TaskGetTool(_TaskToolBase):
    def __init__(self, *, service: TaskService):
        super().__init__(
            service=service,
            name="TaskGet",
            description="Retrieves a task's full record by ID.",
            parameters=TaskGetParams,
            guidance="Inspect state before updating; prefer TaskList to discover multiple tasks.",
            prompt=TASK_GET_PROMPT,
            read_only=True,
            supports_parallel=True,
            source="builtin",
            tags=["task", "read"],
            side_effect_level="none",
            resource_scope=["task", "runtime"],
        )

    def run(self, parameters: dict) -> ToolResult:
        task_id = parameters["task_id"]
        try:
            task = self.service.get_task(task_id)
        except TaskNotFoundError as exc:
            return ToolResult.error(
                str(exc),
                error_type="task_not_found",
                metadata={"task_id": task_id},
            )
        payload = task.model_dump(mode="python")
        return ToolResult.success(
            content=f"任务 {task.task_id}",
            display_text=format_structured_display(
                f"任务 {task.task_id}",
                payload,
            ),
            structured_data=payload,
            metadata={"task_id": task.task_id},
        )


class TaskUpdateTool(_TaskToolBase):
    def __init__(self, *, service: TaskService):
        super().__init__(
            service=service,
            name="TaskUpdate",
            description="Updates an existing task; specify only changed fields.",
            parameters=TaskUpdateParams,
            guidance="Record actual progress. Metadata fields are merged, not replaced wholesale.",
            prompt=TASK_UPDATE_PROMPT,
            read_only=False,
            supports_parallel=False,
            source="builtin",
            tags=["task", "planning"],
            risk_categories=["side_effect"],
            side_effect_level="low",
            resource_scope=["task", "runtime"],
        )

    def run(self, parameters: dict) -> ToolResult:
        update_params = dict(parameters)
        task_id = update_params.pop("task_id")
        try:
            task = self.service.update_task(task_id, **update_params)
        except TaskNotFoundError as exc:
            return ToolResult.error(
                str(exc),
                error_type="task_not_found",
                metadata={"task_id": task_id},
            )
        payload = task.model_dump(mode="python")
        return ToolResult.success(
            content=f"已更新任务 {task.task_id}",
            display_text=format_structured_display(
                f"已更新任务 {task.task_id}",
                payload,
            ),
            structured_data=payload,
            metadata={"task_id": task.task_id},
        )


class TaskListTool(_TaskToolBase):
    def __init__(self, *, service: TaskService):
        super().__init__(
            service=service,
            name="TaskList",
            description="Lists up to 500 tasks, optionally filtered by status, owner, or parent.",
            parameters=TaskListParams,
            guidance="Use parent_task_id for child tasks and TaskGet for a known task.",
            prompt=TASK_LIST_PROMPT,
            read_only=True,
            supports_parallel=True,
            source="builtin",
            tags=["task", "read"],
            side_effect_level="none",
            resource_scope=["task", "runtime"],
        )

    def run(self, parameters: dict) -> ToolResult:
        tasks = self.service.list_tasks(**parameters)
        structured = [task.model_dump(mode="python") for task in tasks]
        return ToolResult.success(
            content=f"共 {len(tasks)} 个任务",
            display_text=f"共 {len(tasks)} 个任务\n{dump_tool_payload(structured)}",
            structured_data=structured,
            metadata={"task_count": len(tasks)},
        )


def register_task_tools(
    registry: ToolRegistry,
    *,
    service: TaskService | None = None,
    expose_in_deferred: bool | None = True,
) -> tuple[TaskCreateTool, TaskGetTool, TaskUpdateTool, TaskListTool]:
    task_service = _service_or_default(service)
    create_tool = TaskCreateTool(service=task_service)
    get_tool = TaskGetTool(service=task_service)
    update_tool = TaskUpdateTool(service=task_service)
    list_tool = TaskListTool(service=task_service)
    registry.register_tools(
        [create_tool, get_tool, update_tool, list_tool],
        expose_in_deferred=expose_in_deferred,
    )
    return create_tool, get_tool, update_tool, list_tool


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "register_task_tools",
]
