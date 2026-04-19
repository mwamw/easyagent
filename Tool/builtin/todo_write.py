"""Claude-style TodoWrite tool."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeTodoItem, ClaudeTodoWriteInput
from ..runtime import TodoItemSnapshot, set_todo_items

if TYPE_CHECKING:
    from task import TaskRecord, TaskService, TaskStatus

TODO_WRITE_PROMPT = """用于维护当前任务的完整 todo 列表。
- 每次调用都要提供完整的 todo 列表，而不是增量 patch。
- `in_progress` 同一时刻最好只有 0 或 1 个。
- `activeForm` 应写成正在进行中的动作，便于模型复述当前步骤。"""

VERIFICATION_KEYWORDS = (
    "verify",
    "verification",
    "test",
    "tests",
    "tested",
    "check",
    "validate",
    "validation",
    "确认",
    "验证",
    "测试",
    "检查",
    "回归",
)

TODO_SCOPE_KEY = "_todo_write.scope_key"
TODO_METADATA_KEY = "_todo_write"


def _normalize_todos(raw_todos: list[dict]) -> list[TodoItemSnapshot]:
    normalized: list[TodoItemSnapshot] = []
    for item in raw_todos:
        todo = ClaudeTodoItem.model_validate(item)
        normalized.append(
            TodoItemSnapshot(
                content=todo.content.strip(),
                status=todo.status,
                active_form=todo.activeForm.strip(),
            )
        )
    return normalized


def _validate_todos(todos: list[TodoItemSnapshot]) -> str | None:
    seen_contents: set[str] = set()
    in_progress_count = 0

    for index, todo in enumerate(todos, start=1):
        if not todo.content:
            return f"错误：第 {index} 个 todo 的 content 不能为空。"
        if not todo.active_form:
            return f"错误：第 {index} 个 todo 的 activeForm 不能为空。"
        if todo.content in seen_contents:
            return f"错误：todo content 不能重复：{todo.content}"
        seen_contents.add(todo.content)
        if todo.status == "in_progress":
            in_progress_count += 1

    if in_progress_count > 1:
        return "错误：同一时间最多只能有一个 in_progress todo。"

    return None


def _todo_lookup(todos: list[TodoItemSnapshot]) -> dict[str, TodoItemSnapshot]:
    return {todo.content: todo for todo in todos}


def _has_verification_todo(todos: list[TodoItemSnapshot]) -> bool:
    for todo in todos:
        haystack = f"{todo.content} {todo.active_form}".lower()
        if any(keyword in haystack for keyword in VERIFICATION_KEYWORDS):
            return True
    return False


def _build_change_summary(old_todos: list[TodoItemSnapshot], new_todos: list[TodoItemSnapshot]) -> dict[str, int]:
    old_lookup = _todo_lookup(old_todos)
    new_lookup = _todo_lookup(new_todos)

    added = sum(1 for content in new_lookup if content not in old_lookup)
    removed = sum(1 for content in old_lookup if content not in new_lookup)
    updated = sum(
        1
        for content, todo in new_lookup.items()
        if content in old_lookup and old_lookup[content] != todo
    )
    unchanged = sum(
        1
        for content, todo in new_lookup.items()
        if content in old_lookup and old_lookup[content] == todo
    )

    return {
        "added": added,
        "removed": removed,
        "updated": updated,
        "unchanged": unchanged,
    }


def _build_status_counts(todos: list[TodoItemSnapshot]) -> dict[str, int]:
    counts = Counter(todo.status for todo in todos)
    return {
        "pending": counts.get("pending", 0),
        "in_progress": counts.get("in_progress", 0),
        "completed": counts.get("completed", 0),
    }


def _task_status_from_todo_status(status: str) -> "TaskStatus":
    from task import TaskStatus

    mapping = {
        "pending": TaskStatus.OPEN,
        "in_progress": TaskStatus.IN_PROGRESS,
        "completed": TaskStatus.COMPLETED,
    }
    return mapping.get(status, TaskStatus.OPEN)


def _todo_status_from_task_status(status: "TaskStatus") -> str:
    value = getattr(status, "value", status)
    mapping = {
        "open": "pending",
        "blocked": "pending",
        "cancelled": "pending",
        "in_progress": "in_progress",
        "completed": "completed",
    }
    return mapping.get(str(value), "pending")


def _todo_snapshot_from_task(task: "TaskRecord") -> TodoItemSnapshot:
    todo_meta = dict(task.metadata.get(TODO_METADATA_KEY) or {})
    active_form = str(todo_meta.get("active_form") or task.description or f"正在处理 {task.title}").strip()
    return TodoItemSnapshot(
        content=task.title.strip(),
        status=_todo_status_from_task_status(task.status),
        active_form=active_form,
    )


def _sort_task_records_for_todo(records: list["TaskRecord"]) -> list["TaskRecord"]:
    return sorted(
        list(records or []),
        key=lambda task: (
            int((task.metadata.get(TODO_METADATA_KEY) or {}).get("order", 0)),
            task.created_at,
        ),
    )


def _sync_with_task_service(
    *,
    service: "TaskService",
    scope_key: str,
    owner: str | None,
    todos: list[TodoItemSnapshot],
) -> tuple[list[TodoItemSnapshot], list[TodoItemSnapshot], list[str]]:
    existing = service.list_tasks(
        metadata_filters={TODO_SCOPE_KEY: scope_key},
        limit=1000,
    )
    visible_existing = [
        task for task in existing
        if bool((task.metadata.get(TODO_METADATA_KEY) or {}).get("visible", True))
    ]
    existing_by_title = {task.title: task for task in existing}
    old_todos = [_todo_snapshot_from_task(task) for task in _sort_task_records_for_todo(visible_existing)]

    seen_titles: set[str] = set()
    synced_records: list["TaskRecord"] = []
    for index, todo in enumerate(todos):
        seen_titles.add(todo.content)
        existing_task = existing_by_title.get(todo.content)
        next_metadata = dict(existing_task.metadata) if existing_task is not None else {}
        todo_meta = dict(next_metadata.get(TODO_METADATA_KEY) or {})
        todo_meta.update(
            {
                "scope_key": scope_key,
                "visible": True,
                "order": index,
                "active_form": todo.active_form,
                "managed_by": "TodoWrite",
                "updated_at": datetime.now().isoformat(),
            }
        )
        next_metadata[TODO_METADATA_KEY] = todo_meta
        if existing_task is None:
            synced_records.append(
                service.create_task(
                    title=todo.content,
                    description=todo.active_form,
                    status=_task_status_from_todo_status(todo.status),
                    owner=owner,
                    metadata=next_metadata,
                )
            )
            continue
        synced_records.append(
            service.update_task(
                existing_task.task_id,
                description=todo.active_form,
                status=_task_status_from_todo_status(todo.status),
                owner=owner if owner is not None else existing_task.owner,
                metadata=next_metadata,
                merge_metadata=False,
            )
        )

    for stale_task in existing:
        if stale_task.title in seen_titles:
            continue
        stale_metadata = dict(stale_task.metadata)
        stale_todo_meta = dict(stale_metadata.get(TODO_METADATA_KEY) or {})
        stale_todo_meta.update(
            {
                "scope_key": scope_key,
                "visible": False,
                "removed_from_view_at": datetime.now().isoformat(),
            }
        )
        stale_metadata[TODO_METADATA_KEY] = stale_todo_meta
        service.update_task(
            stale_task.task_id,
            metadata=stale_metadata,
            merge_metadata=False,
        )

    new_todos = [_todo_snapshot_from_task(task) for task in _sort_task_records_for_todo(synced_records)]
    return old_todos, new_todos, [task.task_id for task in synced_records]


def _format_display_text(
    new_todos: list[TodoItemSnapshot],
    *,
    change_summary: dict[str, int],
    status_counts: dict[str, int],
    verification_nudge_needed: bool,
) -> str:
    lines = [
        (
            "Todo 列表已更新："
            f" {len(new_todos)} 项"
            f"（pending {status_counts['pending']} /"
            f" in_progress {status_counts['in_progress']} /"
            f" completed {status_counts['completed']}）"
        ),
        (
            "变更摘要："
            f" 新增 {change_summary['added']}，"
            f" 更新 {change_summary['updated']}，"
            f" 删除 {change_summary['removed']}，"
            f" 未变 {change_summary['unchanged']}"
        ),
    ]

    if new_todos:
        lines.append("")
        lines.append("当前 todos:")
        for index, todo in enumerate(new_todos, start=1):
            lines.append(f"{index}. [{todo.status}] {todo.content} -> {todo.active_form}")

    if verification_nudge_needed:
        lines.append("")
        lines.append("提示：当前列表里还没有明显的验证/测试步骤，完成代码修改后应补上验证。")

    return "\n".join(lines).strip()


class TodoWriteTool(Tool):
    def __init__(
        self,
        *,
        service: "TaskService | None" = None,
        scope_key: str = "todo_write_default",
        owner: str | None = None,
    ):
        super().__init__(
            name="TodoWrite",
            description="维护当前任务的完整 todo 列表，并返回更新前后的状态。",
            parameters=ClaudeTodoWriteInput,
            guidance="每次传入完整 todo 列表；适合把任务拆成 pending / in_progress / completed 三种状态。",
            read_only=False,
            supports_parallel=False,
            source="builtin",
            prompt=TODO_WRITE_PROMPT,
            tags=["planning", "todo", "claude_code"],
        )
        self.service = service
        self.scope_key = str(scope_key).strip() or "todo_write_default"
        self.owner = owner

    def run(self, parameters: dict) -> ToolResult:
        todos = _normalize_todos(parameters.get("todos", []))
        validation_error = _validate_todos(todos)
        if validation_error:
            return ToolResult.error(validation_error, error_type="invalid_parameters")

        task_ids: list[str] = []
        if self.service is not None:
            old_todos, new_todos, task_ids = _sync_with_task_service(
                service=self.service,
                scope_key=self.scope_key,
                owner=self.owner,
                todos=todos,
            )
            set_todo_items(new_todos)
        else:
            old_todos, new_todos = set_todo_items(todos)
        change_summary = _build_change_summary(old_todos, new_todos)
        status_counts = _build_status_counts(new_todos)
        verification_nudge_needed = (
            status_counts["completed"] > 0 and not _has_verification_todo(new_todos)
        )

        structured_data = {
            "oldTodos": [todo.to_dict() for todo in old_todos],
            "newTodos": [todo.to_dict() for todo in new_todos],
            "verificationNudgeNeeded": verification_nudge_needed,
            "taskBacked": self.service is not None,
            "scopeKey": self.scope_key,
            "taskIds": task_ids,
            "summary": {
                "total": len(new_todos),
                "statusCounts": status_counts,
                "changes": change_summary,
            },
        }

        return ToolResult.success(
            content="Todo 列表已更新。" if self.service is None else "Todo 视图已同步到结构化任务系统。",
            display_text=_format_display_text(
                new_todos,
                change_summary=change_summary,
                status_counts=status_counts,
                verification_nudge_needed=verification_nudge_needed,
            ),
            structured_data=structured_data,
        )


def register_todo_write_tool(
    registry: ToolRegistry,
    *,
    tool: TodoWriteTool | None = None,
    service: "TaskService | None" = None,
    scope_key: str = "todo_write_default",
    owner: str | None = None,
) -> TodoWriteTool:
    registered = tool or TodoWriteTool(service=service, scope_key=scope_key, owner=owner)
    registry.register_tool(registered)
    return registered


__all__ = ["TodoWriteTool", "register_todo_write_tool"]
