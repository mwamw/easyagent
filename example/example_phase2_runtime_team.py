import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv("/home/wxd/LLM/EasyAgent/example/.env")
example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")

from agent import BasicAgent
from core import enable_logging
from core.Config import Config
from core.llm import EasyLLM
from runtime import TeamManager
from task import SQLiteTaskStore, TaskService
from Tool import ToolRegistry
from Tool.builtin import (
    register_agent_tool,
    register_file_edit_tool,
    register_file_write_tool,
    register_filesystem_tools,
    register_send_message_tool,
    register_shell_tools,
    register_team_create_tool,
    register_team_delete_tool,
)


# enable_logging()


async def main() -> None:
    workspace = project_root
    llm = EasyLLM(
        provider="openai",
        model="gemini-3.1-pro-preview"
    )
    config = Config(
        workspace_root=workspace,
        allowed_roots=[workspace],
        enable_worktree=True,
        max_background_tasks=4,
        command_timeout_ms=120000,
    )
    task_service = TaskService(
        SQLiteTaskStore(os.path.join(example_dir, "db", "phase2_runtime_tasks.db"))
    )

    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=workspace)
    register_file_write_tool(registry, workspace_root=workspace)
    register_file_edit_tool(registry, workspace_root=workspace)
    register_shell_tools(registry, workspace_root=workspace)

    agent = BasicAgent(
        name="Phase2Manager",
        llm=llm,
        tool_registry=registry,
        reasoning={"effort":"high"},
        enable_tool=True,
        config=config,
        task_service=task_service,
        verbose_thinking=True,
    )

    agent_tool = register_agent_tool(
        registry,
        parent_agent=agent,
        workspace_root=workspace,
        allowed_roots=(workspace,),
        storage_dir=os.path.join(example_dir, ".phase2-agents"),
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
    agent.bind_runtime(agent_runtime=agent_tool.agent_runtime, team_manager=team_manager)

    query = """
你是一个负责推进 EasyAgent Phase 2 的技术负责人。

请按下面流程工作：
1. 先创建一个名为 `phase2-core` 的团队，描述为“Phase 2 runtime and collaboration team”。
2. 使用 Agent 工具启动两个子 agent，并都设置 `team_name="phase2-core"`：
   - `runtime-auditor`：只读分析 `runtime/`、`Tool/builtin/agent_tool.py`、`core/agent.py`，总结现在多 agent runtime 已经具备哪些能力、还缺什么。
   - `tests-auditor`：只读分析 `test/` 目录，列出还需要哪些协作运行时测试。
   - 后台运行
3. 用 SendMessage 向 `phase2-core` 团队广播一条结构化消息：
   “所有成员先给结论，再补理由；禁止修改文件；只做现状分析。”
4. 如果子 agent 需要隔离环境，可以使用 `isolation="worktree"`。
5. 最后你要汇总：
   - 团队 ID
   - 两个子 agent 的 agentId
   - 每个子 agent 的 outputFile
   - mailbox 广播是否送达
   - 你对当前框架阶段性变化的总结

注意：
- 这个流程要真实使用 TeamCreate、Agent、SendMessage 三类工具。
- 不要假设团队或 agent 已经存在。
- 优先使用结构化 tool 返回值中的 ID 和 outputFile。
"""

    print("=== Phase 2 Runtime Team Example ===")
    print("This example is intentionally not auto-tested. Run it manually while debugging.\n")
    await agent.astream_invoke(query,max_iter=20)
    # agent.invoke(query)

if __name__ == "__main__":
    asyncio.run(main())
    # main()
