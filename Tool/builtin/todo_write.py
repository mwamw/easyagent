"""Claude-style TodoWrite tool."""

from __future__ import annotations

from collections import Counter

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeTodoItem, ClaudeTodoWriteInput
from ..runtime import TodoItemSnapshot, set_todo_items

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
    def __init__(self):
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

    def run(self, parameters: dict) -> ToolResult:
        todos = _normalize_todos(parameters.get("todos", []))
        validation_error = _validate_todos(todos)
        if validation_error:
            return ToolResult.error(validation_error, error_type="invalid_parameters")

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
            "summary": {
                "total": len(new_todos),
                "statusCounts": status_counts,
                "changes": change_summary,
            },
        }

        return ToolResult.success(
            content="Todo 列表已更新。",
            display_text=_format_display_text(
                new_todos,
                change_summary=change_summary,
                status_counts=status_counts,
                verification_nudge_needed=verification_nudge_needed,
            ),
            structured_data=structured_data,
        )


def register_todo_write_tool(registry: ToolRegistry, *, tool: TodoWriteTool | None = None) -> TodoWriteTool:
    registered = tool or TodoWriteTool()
    registry.register_tool(registered)
    return registered


__all__ = ["TodoWriteTool", "register_todo_write_tool"]
