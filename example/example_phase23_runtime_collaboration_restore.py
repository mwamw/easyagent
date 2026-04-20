import os
import sys


example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import BasicAgent
from core.Config import Config
from core.llm import EasyLLM
from runtime import TeamManager
from task import SQLiteTaskStore, TaskService
from Tool import ToolRegistry
from Tool.builtin import (
    register_agent_tool,
    register_filesystem_tools,
    register_send_message_tool,
    register_team_create_tool,
    register_team_delete_tool,
)


def build_manager_agent(workspace: str) -> tuple[BasicAgent, str, str]:
    llm = EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )
    config = Config(
        workspace_root=workspace,
        allowed_roots=[workspace],
        enable_worktree=True,
        max_background_tasks=4,
        command_timeout_ms=120000,
    )
    session_db_path = os.path.join(example_dir, "db", "phase23_runtime_session.db")
    task_db_path = os.path.join(example_dir, "db", "phase23_runtime_tasks.db")
    task_service = TaskService(SQLiteTaskStore(task_db_path))

    registry = ToolRegistry()
    register_filesystem_tools(
        registry,
        workspace_root=workspace,
        allowed_roots=(workspace,),
        cwd=workspace,
    )

    agent = BasicAgent(
        name="Phase23Manager",
        llm=llm,
        enable_tool=True,
        tool_registry=registry,
        config=config,
        task_service=task_service,
        verbose_thinking=True,
        reasoning={"effort": "high"},
    )

    agent_tool = register_agent_tool(
        registry,
        parent_agent=agent,
        workspace_root=workspace,
        allowed_roots=(workspace,),
        storage_dir=os.path.join(example_dir, ".phase23-agents"),
        max_background_tasks=4,
    )
    team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
    agent_tool.agent_runtime.bind_team_manager(team_manager)
    register_send_message_tool(
        registry,
        agent_runtime=agent_tool.agent_runtime,
        parent_agent=agent,
    )
    register_team_create_tool(registry, team_manager=team_manager)
    register_team_delete_tool(registry, team_manager=team_manager)
    agent.bind_runtime(
        agent_runtime=agent_tool.agent_runtime,
        team_manager=team_manager,
    )

    root_task = task_service.create_task(
        title="Phase 2/3 runtime collaboration restore",
        description="让 runtime/team/mailbox 状态可随 session 恢复",
        owner=agent.name,
        metadata={"stage": "phase23"},
    )
    agent.set_current_task(root_task.task_id)
    return agent, session_db_path, root_task.task_id


def main() -> None:
    workspace = project_root
    agent, session_db_path, root_task_id = build_manager_agent(workspace)

    query = f"""
你是 EasyAgent 的框架负责人。请真实使用工具完成下面流程：

1. 创建团队 `restore-reviewers`，描述为“runtime restore validation team”。
2. 使用 Agent 工具启动一个子 agent：
   - description: `review runtime state`
   - team_name: `restore-reviewers`
   - prompt: 只读分析 `runtime/`、`core/agent.py`，总结目前 session restore 已经能保留哪些 runtime 协作状态。
3. 使用 SendMessage 向整个团队广播：
   “你们的结论必须先列出恢复后还能查询到的状态，再补理由；不要修改任何文件。”
4. 最终请输出：
   - teamId
   - agentId
   - outputFile
   - mailbox 是否已送达
   - 子 agent 是否仍绑定 currentTaskId={root_task_id}

注意：
- 必须真实使用 TeamCreate、Agent、SendMessage。
- 这是一次 session restore 演示，不要假设 team 或 agent 已经存在。
"""

    print("=== Phase 2/3 Runtime Collaboration Restore Example ===")
    print("This script is intended for manual debugging and is not auto-executed.")
    print(f"Workspace: {workspace}")
    print(f"Session DB: {session_db_path}")
    print()

    result = agent.invoke(query, max_iter=16)
    print("=== First Run Result ===")
    print(result)
    print()

    session_id = "phase23-runtime-collaboration-demo"
    agent.save_session(session_id, store=session_db_path)
    print(f"Session saved: {session_id}")
    print()

    restored = BasicAgent.load_session(
        session_id,
        llm=EasyLLM(
            provider="openai",
            base_url="http://127.0.0.1:5124/v1",
            api_key="122",
            model="qwen3.5-9b",
        ),
        store=session_db_path,
        task_service=agent.task_service,
    )

    print("=== Restored Runtime State ===")
    print(f"Restored current task: {restored.current_task_id}")
    print(f"Restored teams: {[team.to_dict() for team in restored.team_manager.list_teams()]}")
    print(f"Restored handles: {[handle.to_dict() for handle in restored.agent_runtime.list_handles()]}")
    print()
    print("Next manual checks:")
    print("1. Verify the restored handle still carries the original currentTaskId.")
    print("2. Verify mailbox messages are still present after load_session().")
    print("3. Verify TeamCreate/SendMessage/Agent are still registered on the restored agent.")


if __name__ == "__main__":
    main()
