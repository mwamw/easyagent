from logging import root
import os
import sys


example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")


from agent import BasicAgent
from core.Config import Config
from core.llm import EasyLLM
from runtime import TeamManager
from task import SQLiteTaskStore, TaskService
from Tool import ToolRegistry
from Tool.builtin import (
    register_agent_runtime_tools,
    register_agent_tool,
    register_filesystem_tools,
    register_send_message_tool,
    register_team_create_tool,
    register_team_delete_tool,
)

from core import enable_logging
enable_logging()


def show_result(label: str, result) -> None:
    print(f"=== {label} / display ===")
    print(result.to_display_string())
    print()
    print(f"=== {label} / structured_data ===")
    print(result.structured_data)
    print()


def build_manager() -> tuple[BasicAgent, dict[str, object]]:
    workspace = os.path.join(project_root,"example")
    print(workspace)
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
    task_service = TaskService(
        SQLiteTaskStore(os.path.join(example_dir, "db", "phase23_runtime_final_tasks.db"))
    )

    registry = ToolRegistry()
    register_filesystem_tools(
        registry,
        workspace_root=workspace,
        allowed_roots=(workspace,),
        cwd=workspace,
    )

    agent = BasicAgent(
        name="Phase23FinalManager",
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
        storage_dir=os.path.join(example_dir, ".phase23-final-agents"),
        max_background_tasks=4,
    )
    team_manager = TeamManager(agent_runtime=agent_tool.agent_runtime)
    agent_tool.agent_runtime.bind_team_manager(team_manager)
    register_agent_runtime_tools(
        registry,
        agent_runtime=agent_tool.agent_runtime,
        parent_agent=agent,
    )
    register_send_message_tool(
        registry,
        agent_runtime=agent_tool.agent_runtime,
        parent_agent=agent,
    )
    register_team_create_tool(
        registry,
        team_manager=team_manager,
        parent_agent=agent,
    )
    register_team_delete_tool(
        registry,
        team_manager=team_manager,
    )
    agent.bind_runtime(
        agent_runtime=agent_tool.agent_runtime,
        team_manager=team_manager,
    )

    root_task = task_service.create_task(
        title="Phase 2/3 final runtime completion",
        description="验证 runtime lifecycle、task-agent-team 绑定、session restore",
        owner=agent.name,
        metadata={"stage": "phase23-final"},
    )
    agent.set_current_task(root_task.task_id)

    return agent, {
        "root_task_id": root_task.task_id,
        "workspace": workspace,
        "session_db": os.path.join(example_dir, "db", "phase23_runtime_final_session.db"),
    }


def main() -> None:
    agent, env = build_manager()
    registry = agent.tool_registry
    assert registry is not None

    print("=== Phase 2/3 Final Runtime Example ===")
    print("This example is intended for manual debugging and is not auto-executed.")
    print(f"Workspace: {env['workspace']}")
    print(f"Root task: {env['root_task_id']}")
    print()

    team_result = registry.execute_tool_result(
        "TeamCreate",
        {
            "name": "phase23-final",
            "description": "Runtime lifecycle and collaboration validation team",
        },
    )
    team_id = team_result.structured_data["teamId"]
    show_result("TeamCreate", team_result)

    launch_result = registry.execute_tool_result(
        "Agent",
        {
            "description": "runtime lifecycle audit",
            "prompt": (
                "查看一下agent_test.py这个文件告诉我它的内容"
            ),
            "name": "runtime-lifecycle-auditor",
            "team_name": "phase23-final",
            "run_in_background": True,
        },
    )
    child_agent_id = launch_result.structured_data["agentId"]
    child_task_id = launch_result.metadata.get("taskId")
    show_result("Agent (background)", launch_result)
    print(f"Child task id: {child_task_id}")
    print()

    list_result = registry.execute_tool_result(
        "AgentList",
        {
            "team_id": team_id,
            "limit": 20,
        },
    )
    show_result("AgentList", list_result)

    # team_message_result = registry.execute_tool_result(
    #     "SendMessage",
    #     {
    #         "recipient_type": "team",
    #         "recipient_id": team_id,
    #         "content": "先列恢复后还能查询到的 runtime 状态，再补理由；不要修改任何文件。",
    #     },
    # )
    # show_result("SendMessage(team)", team_message_result)

    if child_task_id:
        task_message_result = registry.execute_tool_result(
            "SendMessage",
            {
                "recipient_type": "task",
                "recipient_id": child_task_id,
                "content": "再查看一下example_stream.py这个文件告诉我它的内容",
            },
        )
        show_result("SendMessage(task)", task_message_result)

    wait_result = registry.execute_tool_result(
        "AgentWait",
        {
            "agent_id": child_agent_id,
            "timeout_ms": 120000,
        },
    )
    show_result("AgentWait", wait_result)

    # second_launch_result = registry.execute_tool_result(
    #     "Agent",
    #     {
    #         "description": "secondary background auditor",
    #         "prompt": (
    #             "做一个尽量完整的只读仓库梳理：先看 runtime/、task/、Tool/builtin/，"
    #             "再总结还剩哪些生命周期 hardening。"
    #         ),
    #         "name": "background-auditor-2",
    #         "team_name": "phase23-final",
    #         "run_in_background": True,
    #     },
    # )
    # second_agent_id = second_launch_result.structured_data["agentId"]
    # show_result("Agent (second background)", second_launch_result)

    # stop_result = registry.execute_tool_result(
    #     "AgentStop",
    #     {
    #         "agent_id": second_agent_id,
    #         "reason": "manual example requests stop after lifecycle validation",
    #         "wait": True,
    #         "timeout_ms": 10000,
    #     },
    # )
    # show_result("AgentStop", stop_result)

    # session_id = "phase23-runtime-collaboration-final"
    # agent.save_session(session_id, store=env["session_db"])
    # print(f"Session saved: {session_id}")
    # print()

    # restored = BasicAgent.load_session(
    #     session_id,
    #     llm=EasyLLM(
    #         provider="openai",
    #         base_url="http://127.0.0.1:5124/v1",
    #         api_key="122",
    #         model="qwen3.5-9b",
    #     ),
    #     store=env["session_db"],
    #     task_service=agent.task_service,
    # )
    # print("=== Restored Runtime State ===")
    # print(f"Current task: {restored.current_task_id}")
    # print(f"Tools: {restored.tool_registry.get_tool_names()}")
    # print(f"Teams: {[team.to_dict() for team in restored.team_manager.list_teams()]}")
    # print(f"Handles: {[handle.to_dict() for handle in restored.agent_runtime.list_handles(limit=20)]}")

    while 1:
        pass
if __name__ == "__main__":
    main()
