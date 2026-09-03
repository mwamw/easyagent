"""Anthropic native provider with directory-based Skills."""

from pathlib import Path

from easyagent import BasicAgent, EasyLLM, ToolRegistry
from Tool.builtin import register_filesystem_tools


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    llm = EasyLLM(
        provider="anthropic_native",
        base_url="http://127.0.0.1:5124",
        api_key="122",
        model="qwen3.5-9b",
    )
    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=str(ROOT))
    agent = BasicAgent("anthropic-skill", llm).with_tool(registry).with_skill(
        Path(__file__).parent / "skills"
    )
    print(agent.invoke("Use repository-review to review skill/manager.py."))


if __name__ == "__main__":
    main()
