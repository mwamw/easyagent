"""Real LLM flow for a path-conditional Skill. Run manually."""

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
        BasicAgent(name="conditional-skill-agent", llm=llm)
        .with_tool(tools)
        .with_skill(SKILLS_DIR)
    )

    # python-module-review is initially hidden. FileRead/Glob/Grep on skill/*.py
    # activates it through tool.invoke.completed, so a later model round can load it.
    answer = agent.invoke(
        "先读取 skill/manager.py，再检查更新后的 available skills。"
        "如果出现 python-module-review，请调用它并用其流程审查该文件。"
    )
    print(answer)
    agent.close()


if __name__ == "__main__":
    main()
