import os
import sys
import time


example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")


from agent import BasicAgent
from core import enable_logging
from core.Config import Config
from core.llm import EasyLLM
from task import SQLiteTaskStore, TaskService
from Tool import ToolRegistry
from Tool.builtin import register_filesystem_tools


enable_logging()


def show_result(label: str, result) -> None:
    print(f"=== {label} / display ===")
    print(result.to_display_string())
    print()
    print(f"=== {label} / structured_data ===")
    print(result.structured_data)
    print()


def build_manager() -> tuple[BasicAgent, dict[str, object]]:
    workspace = os.path.join(project_root, "example")
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
        SQLiteTaskStore(os.path.join(example_dir, "db", "phase23_mailbox_complete_tasks.db"))
    )

    registry = ToolRegistry()
    register_filesystem_tools(
        registry,
        workspace_root=workspace,
        allowed_roots=(workspace,),
        cwd=workspace,
    )

    agent = BasicAgent(
        name="Phase23MailboxManager",
        llm=llm,
        config=config,
    ).with_tool(registry).with_task_service(task_service).with_multi_agent(
        workspace_root=workspace,
        storage_dir=os.path.join(example_dir, ".phase23-mailbox-agents"),
        max_background_tasks=4,
    )
    agent.reasoning = {"effort": "high"}

    root_task = task_service.create_task(
        title="Phase 2/3 mailbox collaboration completion",
        description="验证 mailbox read/ack、自动注入 prompt、completion records",
        owner=agent.name,
        metadata={"stage": "phase23-mailbox-complete"},
    )
    agent.set_current_task(root_task.task_id)

    return agent, {
        "root_task_id": root_task.task_id,
        "workspace": workspace,
    }


def main() -> None:
    agent, env = build_manager()
    registry = agent.tool_registry
    runtime = agent.agent_runtime
    assert registry is not None
    assert runtime is not None

    print("=== Phase 2/3 Mailbox Collaboration Complete Example ===")
    print("This example is for manual debugging. It is not auto-executed by tests.")
    print(f"Workspace: {env['workspace']}")
    print(f"Root task: {env['root_task_id']}")
    print()

    team_result = registry.execute_tool_result(
        "TeamCreate",
        {
            "name": "mailbox-reviewers",
            "description": "Workers that receive mailbox updates during analysis",
        },
    )
    # show_result("TeamCreate", team_result)
    team_id = team_result.structured_data["teamId"]

    launch_result = registry.execute_tool_result(
        "Agent",
        {
            "description": "mailbox-aware reviewer",
            "prompt": (
                "你是一个只读代码审查 worker。\n"
                "先读取 `agent_test.py` 的主要内容并整理一个初步结论。\n"
                "在继续之前，检查协作邮箱：如果 manager 追加了新要求，就调用 MailboxRead 读取完整消息，"
                "按消息里的要求继续执行，并在采用消息后调用 MailboxAck。\n"
                "不要修改任何文件，最后输出简洁的中文总结。"
            ),
            "name": "mailbox-aware-reviewer",
            "team_name": "mailbox-reviewers",
            "run_in_background": True,
        },
    )
    # show_result("Agent (background)", launch_result)
    child_agent_id = launch_result.structured_data["agentId"]

    # 给 worker 一点启动时间，便于它在后续轮次看到 mailbox 更新。
    time.sleep(0.5)

    message_result = registry.execute_tool_result(
        "SendMessage",
        {
            "recipient_type": "agent",
            "recipient_id": child_agent_id,
            "content": (
                "更新任务：在保持只读的前提下，额外读取 `example_stream.py`，"
                "并把它与 `agent_test.py` 的差异一起总结。读取后请确认消费该消息。"
            ),
            "ttl_ms": 300000,
            "metadata": {
                "reason": "manager refined scope",
                "expectedFollowup": "MailboxRead + MailboxAck",
            },
        },
    )
    # show_result("SendMessage(agent)", message_result)

    get_result = registry.execute_tool_result(
        "AgentGet",
        {
            "agent_id": child_agent_id,
        },
    )
    # show_result("AgentGet", get_result)

    manager_mailbox_view = registry.execute_tool_result(
        "MailboxRead",
        {
            "agent_id": child_agent_id,
            "limit": 20,
            "include_consumed": True,
        },
    )
    # show_result("MailboxRead(child mailbox)", manager_mailbox_view)

    wait_result = registry.execute_tool_result(
        "AgentWait",
        {
            "agent_id": child_agent_id,
            "timeout_ms": 120000,
        },
    )
    # show_result("AgentWait", wait_result)

    post_wait_mailbox = registry.execute_tool_result(
        "MailboxRead",
        {
            "agent_id": child_agent_id,
            "limit": 20,
            "include_consumed": True,
            "include_expired": True,
        },
    )
    # show_result("MailboxRead(after wait)", post_wait_mailbox)

    print("=== Runtime completion records ===")
    print([record.to_dict() for record in runtime.list_completion_records()])
    print()

    delete_result = registry.execute_tool_result(
        "TeamDelete",
        {
            "team_id": team_id,
        },
    )
    # show_result("TeamDelete", delete_result)

    print("Manual verification suggestions:")
    print("1. Check whether AgentGet/AgentWait return mailbox counters and outputFile.")
    print("2. Check whether MailboxRead(after wait) shows the message as consumed if the worker called MailboxAck.")
    print("3. Check runtime.list_completion_records() for the completed background worker.")


if __name__ == "__main__":
    main()
