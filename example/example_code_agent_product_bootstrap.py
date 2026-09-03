from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easyagent import BasicAgent, Config, EasyLLM, InMemoryObservabilityStore
from easyagent.callbacks import CallbackManager, StreamingCallback
from easyagent.guardrails import build_default_hook_manager
from easyagent.hooks import BaseHook, HookDecision, HookManager
from easyagent.permissions import PermissionContext, PermissionMode
from easyagent.prompting import PromptBlock, SystemPromptComposer
from easyagent.tools import (
    ToolRegistry,
    register_file_edit_tool,
    register_file_write_tool,
    register_filesystem_tools,
    register_shell_tools,
)


class ProductPromptComposer(SystemPromptComposer):
    """Add product prompt blocks without replacing framework defaults."""

    def __init__(self, product_name: str):
        super().__init__()
        self.product_name = product_name

    def build(self, context):
        return [
            PromptBlock(
                name="product_shell",
                content=(
                    f"你运行在 {self.product_name} 产品中。"
                    "在修改代码前先说明计划；如果需要高风险操作，先明确原因和影响范围。"
                ),
                placement="system_reminder",
                order=120,
            )
        ]


class NoDeleteShellHook(BaseHook):
    """Block obviously destructive shell commands."""

    def before_tool_use(self, payload):
        if payload.get("tool_name") != "Bash":
            return None
        parameters = str(payload.get("parameters", {}))
        if "rm " in parameters or "git reset --hard" in parameters:
            return HookDecision.block("产品默认禁止直接删除文件或硬重置仓库。")
        return None


def build_code_agent(workspace_root: str) -> BasicAgent:
    load_dotenv()

    llm = EasyLLM()
    config = Config(
        workspace_root=workspace_root,
        tool_schema_mode="deferred",
        stable_tool_order=True,
        record_cache_breaks=True,
    )

    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=workspace_root, expose_in_deferred=True)
    register_file_edit_tool(registry, workspace_root=workspace_root, expose_in_deferred=False)
    register_file_write_tool(registry, workspace_root=workspace_root, expose_in_deferred=False)
    register_shell_tools(registry, workspace_root=workspace_root, expose_in_deferred=False)

    callback_manager = CallbackManager([StreamingCallback(verbose=True)])
    hook_manager = build_default_hook_manager()
    hook_manager = HookManager([*hook_manager.hooks, NoDeleteShellHook()])

    agent = BasicAgent(
        name="product-code-agent",
        llm=llm,
        config=config,
        system_prompt=(
            "你是一个谨慎的代码助手。"
            "优先理解现有实现，做最小必要改动；如果发现风险，先解释再继续。"
        ),
    )
    agent.with_tool(registry)
    agent.with_prompt(ProductPromptComposer("EasyAgent Demo IDE"))
    agent.with_permissions(context=PermissionContext(mode=PermissionMode.ASK))
    agent.with_callbacks(callback_manager)
    agent.with_hooks(hook_manager)
    agent.with_observability(store=InMemoryObservabilityStore())
    return agent


def main() -> None:
    workspace_root = str(Path.cwd())
    agent = build_code_agent(workspace_root)

    query = (
        "查看当前仓库结构，找到 README 和 config 相关实现，"
        "并告诉我如果要把这个仓库做成 code agent 产品，最先应该关注哪几个模块。"
    )

    print("\n=== EasyAgent Code Agent Product Bootstrap ===\n")
    result = agent.invoke(query)
    print("\n=== Final Result ===\n")
    print(result)
    print("\n=== Observability Summary ===\n")
    assert agent.observability is not None
    print(agent.observability.summary())


if __name__ == "__main__":
    main()
