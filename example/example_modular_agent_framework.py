"""Real-LLM example for the modular EasyAgent architecture.

This file is intentionally not executed by the repository test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

from easyagent import (
    BasicAgent,
    Config,
    EasyLLM,
    ExecutionMode,
    PlanModeConfig,
    PromptBlock,
    SystemPromptComposer,
    TrainingDataFormat,
    TrainingExporter,
)
from easyagent.context import ContextManager
from easyagent.tasks import InMemoryTaskStore, TaskService
from easyagent.tools import (
    ToolRegistry,
    register_file_edit_tool,
    register_filesystem_tools,
    register_shell_tools,
)


def print_stream(agent: BasicAgent, query: str) -> str:
    final = ""
    for event in agent.stream(query, max_iter=12, temperature=0.2):
        print(json.dumps(event.to_dict(), ensure_ascii=False, default=str))
        if event.type.value == "final":
            final = event.content or ""
    return final


def build_agent(workspace: Path) -> BasicAgent:
    runtime_dir = workspace / ".easyagent-example"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    llm = EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )
    config = Config(
        workspace_root=str(workspace),
        allowed_roots=[str(workspace)],
        tool_schema_mode="deferred",
        command_timeout_ms=120_000,
    )

    prompt = SystemPromptComposer(
        blocks=[
            PromptBlock(
                "identity",
                "你是一个谨慎的 EasyAgent 框架维护助手。先读取实际代码，再给出结论。",
                placement="system",
                order=0,
            ),
            PromptBlock(
                "workspace",
                lambda ctx: (
                    f"当前工作区是 {ctx.execution_context.workspace_root}；"
                    f"当前执行模式是 {ctx.execution_context.execution_mode}。"
                ),
                placement="system_reminder",
                order=100,
            ),
        ]
    )

    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=str(workspace))
    register_file_edit_tool(registry, workspace_root=str(workspace))
    register_shell_tools(registry, workspace_root=str(workspace))

    tasks = TaskService(InMemoryTaskStore())
    agent = (
        BasicAgent(
            name="modular-code-agent",
            llm=llm,
            system_prompt="你是 EasyAgent 仓库的代码维护 Agent。",
            description="Demonstrates explicit optional-module composition.",
            config=config,
        )
        .with_prompt(prompt)
        .with_tool(registry)
        .with_context(ContextManager(max_tokens=16_000, auto_history=True))
        .with_plan(config=PlanModeConfig(register_tools=True))
        .with_task_service(tasks)
        .with_codeintel()
        .with_observability(path=str(runtime_dir / "observability.sqlite3"))
        .with_multi_agent(
            workspace_root=str(workspace),
            storage_dir=str(runtime_dir / "agents"),
        )
    )

    return agent


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    output_dir = workspace / ".easyagent-example" / "training"
    agent = build_agent(workspace)

    try:
        root_task = agent.task_service.create_task(
            title="审查模块化 Agent runtime",
            owner=agent.name,
        )
        agent.set_current_task(root_task.task_id)

        agent.enter_plan_mode(allowed_actions=["read files", "query code intelligence", "delegate read-only analysis"])
        print_stream(
            agent,
            "审查 runtime/multi_agent.py 与 metamessage/manager.py 的协作边界，只制定检查计划，不修改文件。",
        )

        if agent.get_execution_mode() is ExecutionMode.PLAN:
            agent.exit_plan_mode()
        print_stream(
            agent,
            "执行刚才的只读检查；必要时启动一个后台子 Agent，并通过 AgentWait 获取结果。最后总结 mailbox 消息如何进入子 Agent。",
        )

        latest = agent.observability.latest()
        if latest is not None:
            agent.observability.annotate(
                {"example": "modular_agent_framework", "reviewed": True},
                invoke_id=latest.invoke_id,
            )

        report = TrainingExporter.from_agent(agent).export(
            output_dir,
            formats=[
                TrainingDataFormat.STEP_SFT,
                TrainingDataFormat.TRACE_SFT,
                TrainingDataFormat.AGENTIC_ROLLOUT,
            ],
        )
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    finally:
        print(json.dumps(agent.close(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
