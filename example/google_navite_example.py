"""Google native provider with directory-based Skills."""

from pathlib import Path

from easyagent import BasicAgent, EasyLLM, ToolRegistry
from Tool.builtin import register_filesystem_tools


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=str(ROOT))
    agent = BasicAgent("google-skill", EasyLLM(provider="google_native"))
    agent.with_tool(registry).with_skill(Path(__file__).parent / "skills")
    print(agent.invoke("Use repository-review to review skill/manager.py."))


if __name__ == "__main__":
    main()
