from __future__ import annotations

from pathlib import Path

from common import (
    BenchCase,
    CaseResult,
    Workspace,
    append_jsonl,
    create_llm,
    default_arg_parser,
    ensure_results_dir,
    result_to_dict,
    summarize_results,
    write_json,
    now_ms,
)
from easyagent import BasicAgent, Config
from easyagent.permissions import PermissionContext, PermissionMode
from easyagent.tools import ToolRegistry, register_file_edit_tool, register_filesystem_tools, register_shell_tools


def build_cases() -> list[BenchCase]:
    return [
        BenchCase(
            case_id="bug_add",
            category="bug_fix",
            expected="tests_pass",
            task="Fix calc.add so the tests pass. Run the tests and keep the patch minimal.",
            metadata={"kind": "add"},
        ),
        BenchCase(
            case_id="bug_factorial",
            category="bug_fix",
            expected="tests_pass",
            task="Fix calc.factorial so the tests pass. Run the tests and keep the patch minimal.",
            metadata={"kind": "factorial"},
        ),
        BenchCase(
            case_id="feature_slugify",
            category="feature",
            expected="tests_pass",
            task="Implement text.slugify so the tests pass. Run the tests and keep the patch minimal.",
            metadata={"kind": "slugify"},
        ),
    ]


def prepare_workspace(workspace: Workspace, kind: str) -> None:
    if kind == "add":
        workspace.write("calc.py", "def add(a, b):\n    return a - b\n")
        workspace.write("test_calc.py", "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    elif kind == "factorial":
        workspace.write(
            "calc.py",
            "def factorial(n):\n    if n <= 1:\n        return 0\n    return n * factorial(n - 1)\n",
        )
        workspace.write("test_calc.py", "from calc import factorial\n\ndef test_factorial():\n    assert factorial(5) == 120\n")
    elif kind == "slugify":
        workspace.write("text.py", "def slugify(value):\n    raise NotImplementedError\n")
        workspace.write(
            "test_text.py",
            "from text import slugify\n\ndef test_slugify():\n    assert slugify('Hello World!') == 'hello-world'\n",
        )
    workspace.write("README.md", "Synthetic local code-agent benchmark task.\n")


def build_agent(workspace: Path, *, tool_schema_mode: str, permission_mode: str) -> BasicAgent:
    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=str(workspace), expose_in_deferred=True)
    register_file_edit_tool(registry, workspace_root=str(workspace), expose_in_deferred=False)
    register_shell_tools(registry, workspace_root=str(workspace), expose_in_deferred=False)
    config = Config(
        workspace_root=str(workspace),
        allowed_roots=[str(workspace)],
        tool_schema_mode=tool_schema_mode,
        command_timeout_ms=120000,
    )
    agent = BasicAgent(
        name="code_agent_bench",
        llm=create_llm(),
        config=config,
        system_prompt="You are a local code agent. Make minimal edits and run targeted tests.",
    )
    return agent.with_tool(registry).with_permissions(
        context=PermissionContext(mode=PermissionMode(permission_mode))
    )


def run_case(case: BenchCase, *, tool_schema_mode: str, permission_mode: str) -> CaseResult:
    start = now_ms()
    with Workspace(prefix=f"ea_{case.case_id}_") as workspace:
        prepare_workspace(workspace, case.metadata["kind"])
        agent = build_agent(workspace.path, tool_schema_mode=tool_schema_mode, permission_mode=permission_mode)
        error = None
        try:
            agent.invoke(case.task, max_iter=20, temperature=0)
        except Exception as exc:
            error = str(exc)
        test_result = workspace.run(["python", "-m", "pytest", "-q"], timeout=60)
        success = test_result.returncode == 0
        trace = getattr(agent, "trace_history", []) or []
        tool_calls = sum(
            1
            for event in trace
            if isinstance(event, dict) and event.get("type") == "tool.invoke.started"
        )
        usage = {}
        try:
            usage = agent.get_context_usage()
        except Exception:
            usage = {}
        return CaseResult(
            case_id=case.case_id,
            category=case.category,
            expected=case.expected,
            observed="tests_pass" if success else "tests_failed",
            success=success,
            duration_ms=now_ms() - start,
            error=error,
            metrics={
                "workspace": str(workspace.path),
                "tool_schema_mode": tool_schema_mode,
                "permission_mode": permission_mode,
                "tool_calls": tool_calls,
                "pytest_returncode": test_result.returncode,
                "pytest_stdout": test_result.stdout[-2000:],
                "pytest_stderr": test_result.stderr[-2000:],
                "context_usage": usage,
            },
        )


def main() -> None:
    parser = default_arg_parser("E4 local code-agent task benchmark")
    parser.add_argument("--tool-schema-mode", choices=["full", "deferred"], default="deferred")
    parser.add_argument("--permission-mode", choices=["default", "accept_edits", "dont_ask", "bypass"], default="bypass")
    args = parser.parse_args()
    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "e4_code_agent_tasks.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    for case in cases:
        for _ in range(max(args.repeat, 1)):
            result = run_case(case, tool_schema_mode=args.tool_schema_mode, permission_mode=args.permission_mode)
            results.append(result)
            append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary["experiment"] = "e4_code_agent_tasks"
    out_path = Path(args.out) if args.out else out_dir / "e4_code_agent_tasks_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
