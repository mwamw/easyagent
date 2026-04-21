import asyncio
import os
import sys


example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")


from agent import BasicAgent
from core import enable_logging
from core.Config import Config
from core.llm import EasyLLM
from Tool import ToolRegistry
from Tool.builtin import register_codeintel_tools, register_filesystem_tools


enable_logging()


def ensure_sample_workspace() -> str:
    workspace = os.path.join(example_dir, "scratch", "phase_d_codeintel_workspace")
    os.makedirs(workspace, exist_ok=True)
    sample_file = os.path.join(workspace, "sample.py")
    with open(sample_file, "w", encoding="utf-8") as handle:
        handle.write(
            "def demo_function():\n"
            "    value = 1\n"
            "    return value\n\n"
            "demo_function()\n"
        )
    return workspace


def build_agent() -> tuple[BasicAgent, str]:
    workspace = ensure_sample_workspace()
    llm = EasyLLM(
        provider="anthropic_native",
        base_url="http://127.0.0.1:5124",
        api_key="122",
        model="qwen3.5-9b",
    )
    registry = ToolRegistry()
    register_filesystem_tools(
        registry,
        workspace_root=workspace,
        allowed_roots=(workspace,),
        cwd=workspace,
    )

    agent = BasicAgent(
        name="PhaseDCodeIntelManager",
        llm=llm,
        enable_tool=True,
        tool_registry=registry,
        config=Config(
            workspace_root=workspace,
            allowed_roots=[workspace],
        ),
        reasoning={"effort": "high"},
    )

    register_codeintel_tools(
        registry,
        parent_agent=agent,
        workspace_root=workspace,
        allowed_roots=(workspace,),
    )
    return agent, workspace


def show_result(label: str, result) -> None:
    print(f"=== {label} / display ===")
    print(result.to_display_string())
    print()
    print(f"=== {label} / structured_data ===")
    print(result.structured_data)
    print()


def main() -> None:
    agent, workspace = build_agent()
    registry = agent.tool_registry
    assert registry is not None

    print("=== Phase D CodeIntel LSP v1 Example ===")
    print("Manual example only. Do not run it inside automated tests.")
    print(f"Workspace: {workspace}")
    print("If your machine has a usable LSP server, the codeintel tools should return symbol/diagnostic results.")
    print("If not, the same tools should return status=unavailable with fallbackTools=[FileRead, Grep, Glob].")
    print()

    status_result = registry.execute_tool_result(
        "CodeIntelStatus",
        {
            "file_path": "sample.py",
        },
    )
    show_result("CodeIntelStatus", status_result)

    document_symbols_result = registry.execute_tool_result(
        "GetDocumentSymbols",
        {
            "file_path": "sample.py",
        },
    )
    show_result("GetDocumentSymbols", document_symbols_result)

    definition_result = registry.execute_tool_result(
        "FindDefinition",
        {
            "file_path": "sample.py",
            "line": 5,
            "column": 2,
        },
    )
    show_result("FindDefinition", definition_result)

    references_result = registry.execute_tool_result(
        "FindReferences",
        {
            "file_path": "sample.py",
            "line": 1,
            "column": 5,
            "include_declaration": True,
        },
    )
    show_result("FindReferences", references_result)

    diagnostics_result = registry.execute_tool_result(
        "GetDiagnostics",
        {
            "file_path": "sample.py",
        },
    )
    show_result("GetDiagnostics", diagnostics_result)

    print("=== Optional agent-driven prompt ===")
    print(
        "When you want to test the real agent loop, you can run something like:\n"
        'agent.invoke("先检查 sample.py 的 codeintel 可用性，再看符号树、定义、引用和 diagnostics，'
        '如果不可用就退回 FileRead/Grep/Glob。")'
    )
    print()

    print("=== Agent Invoke ===")


    close_report = agent.close(close_worktree=False)
    print("=== Close Report ===")
    print(close_report)


if __name__ == "__main__":
    # main()
    agent, workspace = build_agent()

    asyncio.run(agent.astream_invoke("先检查 sample.py 的 codeintel 可用性，再看符号树、定义、引用和 diagnostics，如果不可用就退回 FileRead/Grep/Glob。"))