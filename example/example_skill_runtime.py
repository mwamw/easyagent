"""Real directory-based Skill flow. Run manually when the local LLM is ready."""

from pathlib import Path

from easyagent import BasicAgent, EasyLLM, ToolRegistry
from Tool.builtin import register_filesystem_tools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def main() -> None:
    llm = EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )
    tools = ToolRegistry()
    register_filesystem_tools(
        tools,
        workspace_root=str(PROJECT_ROOT),
        allowed_roots=[str(PROJECT_ROOT)],
    )

    agent = (
        BasicAgent(name="skill-code-review", llm=llm)
        .with_tool(tools)
        .with_skill(SKILLS_DIR)
    )
    result = agent.invoke(
        "使用 repository-review skill 审查 skill/manager.py，重点检查生命周期和权限清理。"
    )
    print(result)
    agent.close()


if __name__ == "__main__":
    main()
