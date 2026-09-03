import os
import subprocess
import sys


example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")


from agent import BasicAgent
from core import enable_logging
from core.Config import Config
from core.llm import EasyLLM
from runtime import MultiAgentRuntime
from task import SQLiteTaskStore, TaskService
from Tool import ToolRegistry
from Tool.builtin import register_filesystem_tools
from Tool.runtime import WorktreeManager


enable_logging()


def ensure_example_repo() -> str:
    repo = os.path.join(example_dir, "scratch", "phasec_restore_repo")
    os.makedirs(repo, exist_ok=True)
    git_dir = os.path.join(repo, ".git")
    if not os.path.isdir(git_dir):
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.email", "phasec@example.com"], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(["git", "config", "user.name", "Phase C Example"], cwd=repo, check=True, capture_output=True, text=True)
        with open(os.path.join(repo, "README.md"), "w", encoding="utf-8") as handle:
            handle.write("# Phase C Example Repo\n")
        with open(os.path.join(repo, "app.py"), "w", encoding="utf-8") as handle:
            handle.write("def main():\n    return 'phase-c'\n")
        subprocess.run(["git", "add", "README.md", "app.py"], cwd=repo, check=True, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", "init phase-c example"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def build_agent() -> tuple[BasicAgent, dict[str, str]]:
    repo = ensure_example_repo()
    session_db = os.path.join(example_dir, "db", "phasec_restore_report_session.db")
    task_db = os.path.join(example_dir, "db", "phasec_restore_report_tasks.db")
    worktree_storage = os.path.join(example_dir, ".phasec-worktrees")
    agent_storage = os.path.join(example_dir, ".phasec-agents")

    llm = EasyLLM(
        provider="anthropic_native",
        base_url="http://127.0.0.1:5124",
        api_key="122",
        model="qwen3.5-9b",
    )
    config = Config(
        workspace_root=repo,
        allowed_roots=[repo],
        enable_worktree=True,
        max_background_tasks=4,
        command_timeout_ms=120000,
    )
    task_service = TaskService(SQLiteTaskStore(task_db))
    registry = ToolRegistry()
    register_filesystem_tools(
        registry,
        workspace_root=repo,
        allowed_roots=(repo,),
        cwd=repo,
    )

    repo_root = WorktreeManager.detect_repo_root(repo)
    worktree_manager = WorktreeManager(
        repo_root=repo_root,
        storage_dir=worktree_storage,
        original_cwd=repo,
    )
    agent = BasicAgent(
        name="PhaseCRestoreManager",
        llm=llm,
        config=config,
    ).with_tool(registry).with_task_service(task_service).with_worktree(
        worktree_manager,
    ).with_multi_agent(
        workspace_root=repo,
        storage_dir=agent_storage,
        max_background_tasks=4,
    )
    agent.reasoning = {"effort": "high"}

    root_task = task_service.create_task(
        title="Phase C restore report walkthrough",
        description="Verify restore report and cleanup report with real LLM-backed subagents",
        owner=agent.name,
        metadata={"stage": "phasec"},
    )
    agent.set_current_task(root_task.task_id)

    return agent, {
        "repo": repo,
        "session_db": session_db,
        "task_db": task_db,
        "root_task_id": root_task.task_id,
    }


def main() -> None:
    agent, env = build_agent()
    registry = agent.tool_registry
    assert registry is not None

    print("=== Phase C Restore Report And Lifecycle Example ===")
    print("Manual example only. Do not run it inside automated tests.")
    print(f"Repo: {env['repo']}")
    print(f"Root task: {env['root_task_id']}")
    print(f"Session DB: {env['session_db']}")
    print()

    team_result = registry.execute_tool_result(
        "TeamCreate",
        {
            "name": "phasec-reviewers",
            "description": "Workers that help validate restore and cleanup semantics",
        },
    )
    print("TeamCreate:", team_result.to_display_string())
    print()

    enter_result = registry.execute_tool_result(
        "EnterWorktree",
        {
            "name": "phasec-restore-check",
        },
    )
    print("EnterWorktree:", enter_result.to_display_string())
    print()

    launch_result = registry.execute_tool_result(
        "Agent",
        {
            "description": "background lifecycle auditor",
            "prompt": (
                "只读分析当前仓库，重点阅读 README.md、app.py、core/agent.py、runtime/agents/manager.py。"
                "给出一个较完整的恢复语义总结，包含 runtime restore、worktree restore、mailbox 和 close report。"
                "不要修改文件。"
            ),
            "name": "phasec-background-auditor",
            "team_name": "phasec-reviewers",
            "run_in_background": True,
        },
    )
    print("Agent(background):", launch_result.to_display_string())
    print()

    session_id = "phasec-restore-report-lifecycle"
    agent.save_session(session_id, store=env["session_db"])
    print(f"Session saved immediately: {session_id}")
    print("If the background agent has not finished yet, restore report should show a degraded/interrupted runtime path.")
    print()

    restored = BasicAgent.load_session(
        session_id,
        llm=EasyLLM(
            provider="openai",
            base_url="http://127.0.0.1:5124/v1",
            api_key="122",
            model="qwen3.5-9b",
        ),
        store=env["session_db"],
        task_service=agent.task_service,
        worktree_manager=WorktreeManager(
            repo_root=WorktreeManager.detect_repo_root(env["repo"]),
            storage_dir=os.path.join(example_dir, ".phasec-worktrees"),
            original_cwd=env["repo"],
        ),
        multi_agent_runtime=MultiAgentRuntime(
            workspace_root=env["repo"],
            storage_dir=os.path.join(example_dir, ".phasec-agents"),
            max_background_tasks=4,
        ),
    )

    print("=== Restore Report ===")
    print(restored.get_last_restore_report())
    print()

    print("=== Restored Agent Handles ===")
    if restored.agent_runtime is not None:
        print([handle.to_dict() for handle in restored.agent_runtime.list_handles(limit=20)])
        print()

    close_report = restored.close(worktree_action="keep", close_llm=True)
    print("=== Close Report ===")
    print(close_report)


if __name__ == "__main__":
    main()
