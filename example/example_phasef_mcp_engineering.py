"""Phase F MCP engineering example.

This file is intentionally not executed by the implementation step.
It is meant for manual debugging after the Phase F code changes.
"""

from __future__ import annotations

import os
import sys
example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")
from agent import BasicAgent
from core.llm import EasyLLM
from mcp import MCPPolicyContext, MCPPolicyRule
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_mcp_tools


def build_llm() -> EasyLLM:
    return EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )


def build_registry(workspace_root: str) -> ToolRegistry:
    registry = ToolRegistry()
    server_script = os.path.join(
        workspace_root,
        "mcp",
        "examples",
        "real_python_mcp_server.py",
    )
    policy = MCPPolicyContext(
        capability_cache_ttl_seconds=300,
        resource_cache_ttl_seconds=60,
        persist_connection=False,
        rules=[
            MCPPolicyRule(
                effect="deny",
                server_names=("demo_server",),
                capability_kinds=("prompt_get",),
                reason="这个 demo 不允许直接拉取远程 prompt。",
            )
        ],
    )

    register_mcp_tools(
        registry,
        server_source=["python", server_script],
        tool_prefix="demo_",
        include_resources=True,
        resource_tool_prefix="demo_",
        server_name="demo_server",
        policy_context=policy,
    )
    return registry


def build_agent(workspace_root: str) -> BasicAgent:
    registry = build_registry(workspace_root)
    return BasicAgent(
        name="phasef-mcp-agent",
        llm=build_llm(),
        system_prompt=(
            "你是一个会使用 MCP 远程能力的 code assistant。\n"
            "先确认当前问题是否真的需要 MCP；如果需要，再严格按 schema 调用。\n"
            "拿到远程结果后，优先引用结构化事实，不要自行改写关键字段。"
        ),
    ).with_tool(registry)


def main() -> None:
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    store_path = os.path.join(workspace_root, "db", "phasef_demo_sessions.db")
    session_id = "phasef-mcp-demo"

    agent = build_agent(workspace_root)

    # 1. 查看 MCP runtime 当前状态与 capability snapshot。
    manager_surfaces = agent.tool_registry.list_runtime_surfaces("mcp_manager")
    manager = manager_surfaces["demo_server"]
    print("=== MCP Connection State ===")
    print(manager.connection_state())
    print("=== MCP Capability Snapshot ===")
    print(manager.snapshot(refresh=True).to_dict())

    # 2. 保存 session，快照里会带上 mcp_runtime。
    agent.save_session(session_id, store=store_path)

    # 3. 重新恢复 session，ToolRegistry 会自动重建 MCP runtime。
    restored = BasicAgent.load_session(
        session_id,
        llm=build_llm(),
        store=store_path,
    )
    print("=== Restore Report ===")
    print(restored.get_last_restore_report())

    # 4. 你后续可以手动执行真实 invoke，例如：
    # restored.invoke("请先查看 MCP 能力，再调用合适的 demo_ 工具完成简单计算。")

    # 5. 关闭时会拿到 MCP runtime close report。
    print("=== Close Report ===")
    print(restored.close(close_llm=False))


if __name__ == "__main__":
    main()
