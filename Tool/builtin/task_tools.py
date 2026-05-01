"""Structured task tools for EasyAgent."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from core.Exception import TaskNotFoundError
from task import InMemoryTaskStore, TaskService, TaskStatus

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from .display_utils import dump_tool_payload, format_structured_display


TASK_CREATE_PROMPT = (
    "在结构化任务系统中创建一条持久化的任务记录。\n"
    "\n"
    "何时使用：\n"
    "- 当你需要把一个复杂目标拆解成多个可独立跟踪的子任务时。\n"
    "- 当你要为子 agent 分配工作，并希望后续能通过 TaskGet / TaskList 查询执行进度时。\n"
    "- 当你需要建立任务之间的父子关系，形成可视化的任务树时。\n"
    "\n"
    "参数说明：\n"
    "- title：简洁、明确的任务标题，后续用于在列表中识别任务。\n"
    "- description：详细的任务描述，包含目标、背景、约束和验收标准。\n"
    "- status：初始状态，通常为 open。只有在任务创建时就已经在执行中的情况下才设为 in_progress。\n"
    "- owner：任务归属者的名称（通常是 agent 名称）。不设置时表示未分配。\n"
    "- parent_task_id：父任务 ID。设置后会形成父子层级关系，便于用 TaskList(parent_task_id=...) 查询子任务。\n"
    "- metadata：附加元数据字典，适合存放阶段标签、来源标识、优先级、关联的 agent_id 等结构化上下文。\n"
    "\n"
    "状态含义：\n"
    "- open：任务已创建，尚未开始执行。\n"
    "- in_progress：任务正在执行中。\n"
    "- blocked：任务因依赖、异常或等待外部输入而阻塞。\n"
    "- completed：任务已成功完成。\n"
    "- cancelled：任务已取消，不再需要执行。\n"
    "\n"
    "与其他工具的协作：\n"
    "- 创建后可通过 TaskUpdate 推进状态。\n"
    "- 当使用 Agent tool 委派子任务时，框架会自动创建 child task，通常不需要手动创建。\n"
    "- TodoWrite 是更轻量的替代方案：它在内部自动创建/更新任务记录，适合不需要精确控制 task_id 的场景。\n"
    "\n"
    "不要这样用：\n"
    "- 不要为每一个微小操作都创建任务；任务系统适合跟踪有明确起止、需要状态流转的工作单元。\n"
    "- 不要创建标题模糊的任务（如\"处理一下\"），标题应能让人不看描述就知道任务在做什么。\n"
    "- 不要在不需要持久化追踪的场景下使用任务系统，简单的一次性操作直接执行即可。"
)


TASK_GET_PROMPT = (
    "读取单个任务的完整结构化记录。\n"
    "\n"
    "何时使用：\n"
    "- 当你已经知道 task_id，需要查看它的最新状态、描述、归属者和元数据时。\n"
    "- 当你需要在更新任务之前确认它的当前状态时。\n"
    "- 当子 agent 完成后，你需要检查关联任务的状态是否已同步时。\n"
    "\n"
    "返回内容：\n"
    "- 完整的 TaskRecord，包含 task_id、title、description、status、owner、parent_task_id、metadata、created_at、updated_at。\n"
    "\n"
    "最佳实践：\n"
    "- 如果你不确定 task_id 是否存在，调用后检查返回结果中是否有 task_not_found 错误。\n"
    "- 如果你需要查看某个父任务下的所有子任务，优先用 TaskList(parent_task_id=...) 而不是逐个 TaskGet。"
)


TASK_UPDATE_PROMPT = (
    "更新一条已有任务的字段或状态。\n"
    "\n"
    "何时使用：\n"
    "- 当任务开始执行时，将状态从 open 更新为 in_progress。\n"
    "- 当任务完成时，将状态更新为 completed。\n"
    "- 当任务遇到阻塞（如依赖未就绪、需要用户确认），将状态更新为 blocked。\n"
    "- 当需要补充或修改任务的描述、归属者或元数据时。\n"
    "\n"
    "参数说明：\n"
    "- 只需传入要更新的字段，未传入的字段保持不变。\n"
    "- metadata：传入的 metadata 会与现有 metadata 合并（增量更新），不会覆盖原有字段。\n"
    "- status：状态更新应遵循合理的流转顺序（open -> in_progress -> completed/blocked），但系统不强制校验。\n"
    "\n"
    "推荐的状态流转：\n"
    "- open -> in_progress：任务开始执行。\n"
    "- in_progress -> completed：任务成功完成。\n"
    "- in_progress -> blocked：任务遇到阻塞。\n"
    "- blocked -> in_progress：阻塞解除，继续执行。\n"
    "- 任意状态 -> cancelled：任务不再需要。\n"
    "\n"
    "与其他工具的协作：\n"
    "- 当 Agent tool 启动的子 agent 完成或失败时，框架会自动更新关联的任务状态，通常不需要手动更新。\n"
    "- 如果你手动管理任务生命周期，应在关键节点（开始、完成、阻塞）及时更新状态。\n"
    "\n"
    "不要这样用：\n"
    "- 不要在每一步操作后都更新任务状态；只在状态真正发生变化时才更新。\n"
    "- 不要用 TaskUpdate 代替 TaskCreate；如果任务不存在，TaskUpdate 会报 task_not_found 错误。"
)


TASK_LIST_PROMPT = (
    "列出任务系统中的任务记录，支持按状态、归属者和父任务过滤。\n"
    "\n"
    "何时使用：\n"
    "- 当你需要了解当前有哪些任务正在进行、哪些已完成、哪些被阻塞时。\n"
    "- 当你需要查看某个父任务下的所有子任务及其进度时。\n"
    "- 当你需要找到某个 agent 负责的所有任务时。\n"
    "\n"
    "过滤参数：\n"
    "- status：按状态过滤（open、in_progress、blocked、completed、cancelled）。\n"
    "- owner：按归属者名称过滤，通常是 agent 名称。\n"
    "- parent_task_id：按父任务 ID 过滤，用于查看子任务列表。\n"
    "- limit：返回数量上限，默认 100，最大 500。\n"
    "\n"
    "最佳实践：\n"
    "- 在拆分任务之前，先用 TaskList 检查是否已有相关任务存在，避免重复创建。\n"
    "- 通过 TaskList(status=\"in_progress\") 快速了解当前正在执行的工作。\n"
    "- 通过 TaskList(parent_task_id=\"xxx\") 检查某个主任务的子任务完成情况，判断是否可以推进到下一阶段。\n"
    "- 如果已知具体的 task_id，优先用 TaskGet 而不是 TaskList 加过滤。"
)



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
            description="在结构化任务系统中创建一条持久化任务记录，支持父子层级和元数据。",
            parameters=TaskCreateParams,
            guidance="适合把复杂目标拆成多个可独立跟踪的子任务。标题应明确、描述应包含目标和验收标准。如果只是维护 todo 列表，优先用 TodoWrite。",
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
            description="读取单个任务的完整结构化记录，包含状态、归属者、元数据和时间戳。",
            parameters=TaskGetParams,
            guidance="适合在更新任务前确认当前状态，或在子 agent 完成后检查关联任务。查看多个任务时优先用 TaskList。",
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
            description="更新已有任务的状态、标题、描述、归属者或元数据，只需传入要变更的字段。",
            parameters=TaskUpdateParams,
            guidance="只在状态真正发生变化时才调用。metadata 为增量合并，不会覆盖原有字段。",
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
            description="列出任务记录，支持按状态、归属者和父任务 ID 过滤，最多返回 500 条。",
            parameters=TaskListParams,
            guidance="适合了解任务全局进度。用 parent_task_id 查看子任务、用 status 过滤进行中的工作。已知具体 task_id 时优先用 TaskGet。",
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
