"""Phase 1A example: permission store + task-backed TodoWrite + session restore.

This example is intentionally not executed automatically.
It uses a real EasyLLM configuration and a real on-disk workspace so you can
debug the current framework changes manually.
"""

from __future__ import annotations

from pathlib import Path

from agent import BasicAgent
from core.Config import Config
from core.llm import EasyLLM
from core.permissions import PermissionBehavior, PermissionRule
from task import SQLiteTaskStore, TaskService
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import (
    register_enter_plan_mode_tool,
    register_exit_plan_mode_tool,
    register_file_edit_tool,
    register_file_read_tool,
    register_file_write_tool,
    register_task_tools,
    register_todo_write_tool,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "phase1a_workspace"
SESSION_DB = WORKSPACE / "db" / "phase1a_sessions.sqlite3"
TASK_DB = WORKSPACE / "db" / "phase1a_tasks.sqlite3"
TARGET_FILE = WORKSPACE / "target.py"
NOTES_FILE = WORKSPACE / "notes.md"


def prepare_workspace() -> None:
    (WORKSPACE / "db").mkdir(parents=True, exist_ok=True)
    if not TARGET_FILE.exists():
        TARGET_FILE.write_text(
            "\n".join(
                [
                    '"""Phase 1A target file."""',
                    "",
                    "def build_plan() -> list[str]:",
                    '    return ["inspect current implementation", "write plan items"]',
                    "",
                    "",
                    "def summarize_changes() -> str:",
                    '    return "Pending update"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if not NOTES_FILE.exists():
        NOTES_FILE.write_text(
            "\n".join(
                [
                    "# Phase 1A Notes",
                    "",
                    "- TodoWrite should sync into TaskService.",
                    "- Workspace-scoped write rules should override generic session prompts.",
                    "- Session restore should preserve context usage and current task state.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def build_agent() -> BasicAgent:
    llm = EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )
    config = Config(
        workspace_root=str(WORKSPACE),
        allowed_roots=[str(WORKSPACE)],
        max_history_length=20,
    )
    task_service = TaskService(SQLiteTaskStore(str(TASK_DB)))

    registry = ToolRegistry()
    register_file_read_tool(
        registry,
        workspace_root=str(WORKSPACE),
        allowed_roots=[str(WORKSPACE)],
        cwd=str(WORKSPACE),
    )
    register_file_edit_tool(
        registry,
        workspace_root=str(WORKSPACE),
        allowed_roots=[str(WORKSPACE)],
        cwd=str(WORKSPACE),
    )
    register_file_write_tool(
        registry,
        workspace_root=str(WORKSPACE),
        allowed_roots=[str(WORKSPACE)],
        cwd=str(WORKSPACE),
    )
    register_task_tools(registry, service=task_service)
    register_todo_write_tool(
        registry,
        service=task_service,
        scope_key="phase1a-demo",
        owner="phase1a-manager",
    )
    register_enter_plan_mode_tool(registry)
    register_exit_plan_mode_tool(registry)

    agent = BasicAgent(
        name="phase1a-manager",
        llm=llm,
        config=config,
        system_prompt=(
            "You are validating Phase 1A of EasyAgent. "
            "Use TodoWrite as a task-backed summary view, respect permission rules, "
            "and explain what state would survive a session restore."
        ),
    ).with_tool(registry).with_task_service(task_service)

    agent.set_permission_rules(
        "workspace_allow",
        [
            PermissionRule(
                tool_name="FileWrite",
                behavior=PermissionBehavior.ALLOW,
                matcher={"path_prefixes": [str(WORKSPACE)]},
                description="工作区内的文件写入允许直接执行。",
            ),
            PermissionRule(
                tool_name="FileEdit",
                behavior=PermissionBehavior.ALLOW,
                matcher={"path_prefixes": [str(WORKSPACE)]},
                description="工作区内的文件编辑允许直接执行。",
            ),
        ],
        priority=10,
    )
    agent.set_permission_rules(
        "session_guard",
        [
            PermissionRule(
                tool_name="FileWrite",
                behavior=PermissionBehavior.ASK,
                description="默认文件写入需要确认。",
            ),
            PermissionRule(
                tool_name="FileEdit",
                behavior=PermissionBehavior.ASK,
                description="默认文件编辑需要确认。",
            ),
        ],
        priority=50,
    )
    return agent


def build_prompt() -> str:
    return f"""
你正在验证 EasyAgent 的 Phase 1A 收口能力，请在这个真实 workspace 中完成一次最小流程。

工作区文件：
- target: {TARGET_FILE}
- notes: {NOTES_FILE}

要求：
1. 先读取 notes 和 target，理解当前状态。
2. 立刻调用 TodoWrite，给出完整 todo 列表。
3. 修改 {TARGET_FILE}：
   - 让 build_plan() 返回 3 个更具体的步骤
   - 让 summarize_changes() 返回一个明确提到 "task-backed todo view" 的字符串
4. 调用 TaskList，确认 TodoWrite 已经同步成结构化任务。
5. 最终说明：
   - 哪些权限规则命中了
   - 创建或更新了哪些任务
   - 如果现在执行 save_session("phase1a-demo") 再 load_session，会保留哪些状态

注意：
- 这个 workspace 内的 FileWrite/FileEdit 已被高优先级规则允许。
- 如果你尝试写到工作区外，低优先级 session_guard 仍然会要求确认。
""".strip()


def main() -> None:
    prepare_workspace()
    agent = build_agent()
    prompt = build_prompt()

    print("Phase 1A example is ready.")
    print("Workspace:", WORKSPACE)
    print("Target file:", TARGET_FILE)
    print("Task DB:", TASK_DB)
    print()
    print("Run manually if you want to inspect the current framework behavior:")
    print()
    print("result = agent.invoke(prompt, max_iter=12)")
    print(f'agent.save_session("phase1a-demo", store="{SESSION_DB}")')
    print(
        "restored = BasicAgent.load_session("
        f'"phase1a-demo", llm=agent.llm, store="{SESSION_DB}", task_service=agent.task_service'
        ")"
    )
    print("print(restored.get_context_usage())")
    print()
    print("Prompt preview:")
    print(prompt)


if __name__ == "__main__":
    main()
