"""Phase G SDK release example.

This file is intentionally not executed by the implementation step.
It demonstrates the stable public SDK entrypoint only.
"""

from __future__ import annotations

import os

from easyagent import BasicAgent, EasyLLM, PermissionBehavior, PermissionContext, PermissionRule, ToolRegistry
from easyagent.session import SessionStore
from easyagent.tools import register_task_tools
from easyagent.tasks import InMemoryTaskStore, TaskService


def build_llm() -> EasyLLM:
    return EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )


def build_agent(workspace_root: str) -> BasicAgent:
    registry = ToolRegistry()
    task_service = TaskService(InMemoryTaskStore())
    register_task_tools(registry, service=task_service)

    permission_context = PermissionContext(
        rules=[
            PermissionRule(
                tool_name="TaskCreate",
                behavior=PermissionBehavior.ALLOW,
                description="允许直接创建结构化任务。",
            ),
            PermissionRule(
                tool_name="TaskUpdate",
                behavior=PermissionBehavior.ALLOW,
                description="允许直接更新结构化任务。",
            ),
        ]
    )

    return BasicAgent(
        name="phaseg-sdk-agent",
        llm=build_llm(),
        tool_registry=registry,
        task_service=task_service,
        permission_context=permission_context,
        enable_tool=True,
        system_prompt=(
            "你是一个通过 EasyAgent 公共 SDK 构建的助手。\n"
            "如果当前问题需要拆解工作，请优先用结构化 task tools。"
        ),
    )


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    store_path = os.path.join(workspace_root, "example", "db", "phaseg_sdk_sessions.db")
    session_id = "phaseg-sdk-demo"

    # 1. 用稳定 SDK 入口构造 agent。
    agent = build_agent(workspace_root)

    # 2. 你后续可以手动调试真实调用，例如：
    # agent.invoke("请创建一个三步的开发任务清单，并解释每步目的。")

    # 3. 保存并恢复 session，观察 restore report。
    agent.save_session(session_id, store=store_path)
    restored = BasicAgent.load_session(
        session_id,
        llm=build_llm(),
        store=SessionStore(store_path),
    )
    print("=== Restore Report ===")
    print(restored.get_last_restore_report())

    # 4. 查看 SDK agent 的 close report。
    print("=== Close Report ===")
    print(restored.close(close_llm=False))


if __name__ == "__main__":
    main()
