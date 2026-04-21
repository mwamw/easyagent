"""Structured task tools for EasyAgent."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from core.Exception import TaskNotFoundError
from task import InMemoryTaskStore, TaskService, TaskStatus

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import dump_tool_payload, format_structured_display


class TaskCreateParams(BaseModel):
    title: str = Field(description="任务标题")
    description: str = Field(default="", description="任务描述")
    status: TaskStatus = Field(default=TaskStatus.OPEN, description="初始状态")
    owner: Optional[str] = Field(default=None, description="任务归属者")
    parent_task_id: Optional[str] = Field(default=None, description="父任务 ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class TaskGetParams(BaseModel):
    task_id: str = Field(description="任务 ID")


class TaskUpdateParams(BaseModel):
    task_id: str = Field(description="任务 ID")
    title: Optional[str] = Field(default=None, description="新标题")
    description: Optional[str] = Field(default=None, description="新描述")
    status: Optional[TaskStatus] = Field(default=None, description="新状态")
    owner: Optional[str] = Field(default=None, description="新归属者")
    parent_task_id: Optional[str] = Field(default=None, description="父任务 ID")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="元数据增量")


class TaskListParams(BaseModel):
    status: Optional[TaskStatus] = Field(default=None, description="状态过滤")
    owner: Optional[str] = Field(default=None, description="归属者过滤")
    parent_task_id: Optional[str] = Field(default=None, description="父任务 ID 过滤")
    limit: int = Field(default=100, ge=1, le=500, description="返回数量上限")


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
            description="创建结构化任务。",
            parameters=TaskCreateParams,
            guidance="适合把长任务拆成可跟踪对象，而不是只写 todo 文本。",
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
            description="获取单个任务详情。",
            parameters=TaskGetParams,
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
            description="更新任务字段或状态。",
            parameters=TaskUpdateParams,
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
            description="列出任务列表。",
            parameters=TaskListParams,
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
) -> tuple[TaskCreateTool, TaskGetTool, TaskUpdateTool, TaskListTool]:
    task_service = _service_or_default(service)
    create_tool = TaskCreateTool(service=task_service)
    get_tool = TaskGetTool(service=task_service)
    update_tool = TaskUpdateTool(service=task_service)
    list_tool = TaskListTool(service=task_service)
    registry.register_tools([create_tool, get_tool, update_tool, list_tool])
    return create_tool, get_tool, update_tool, list_tool


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "register_task_tools",
]
