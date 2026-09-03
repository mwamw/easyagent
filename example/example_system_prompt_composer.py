"""Real-LLM example for the unified system prompt composer.

This file is intentionally excluded from automated tests. Run it manually after
the OpenAI-compatible endpoint is available at 127.0.0.1:5124.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easyagent import BasicAgent, Config, EasyLLM
from easyagent.prompting import PromptBlock, SystemPromptComposer
from easyagent.tools import ToolRegistry, register_filesystem_tools


class CodeProductPromptComposer(SystemPromptComposer):
    """A product composer configured with two ordinary PromptBlock values."""

    def __init__(self) -> None:
        super().__init__(
            blocks=[
                PromptBlock(
                    name="product_policy",
                    content=(
                        "## 产品规则\n"
                        "先检查仓库中的真实文件和实现，再给结论。"
                        "只做当前任务需要的改动，不创建兼容垫片。"
                    ),
                    placement="system",
                    order=110,
                ),
                PromptBlock(
                    name="workspace_context",
                    content=lambda context: (
                        "## 当前产品上下文\n"
                        f"Agent: {context.agent_name}\n"
                        f"Workspace: {context.execution_context.workspace_root}\n"
                        "本段是 request-scoped system reminder，不会写入长期对话历史。"
                    ),
                    placement="system_reminder",
                    order=120,
                    metadata={"cache_partition": "session", "cacheable": True},
                ),
            ]
        )


def build_agent(workspace_root: str) -> BasicAgent:
    llm = EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )
    config = Config(workspace_root=workspace_root)

    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=workspace_root)

    return BasicAgent(
        name="real-code-agent",
        llm=llm,
        config=config,
        system_prompt="你是一个可以检查真实仓库的代码 Agent。",
    ).with_tool(registry).with_prompt(CodeProductPromptComposer())


def main() -> None:
    workspace_root = str(Path.cwd())
    agent = build_agent(workspace_root)

    print("=== Provider-native system prompt ===")
    print(agent.get_enhanced_prompt())
    print("\n=== Request-scoped system reminders ===")
    print(agent.get_system_prompt_template().render_system_reminders())

    result = agent.invoke(
        "读取当前仓库的 README.md，概括这个项目的用途，并指出系统提示词扩展入口。"
    )
    print("\n=== LLM result ===")
    print(result)


if __name__ == "__main__":
    main()
