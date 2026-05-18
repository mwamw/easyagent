from __future__ import annotations

import tempfile
from pathlib import Path

from common import (
    BenchCase,
    CaseResult,
    append_jsonl,
    build_mock_registry,
    build_permission_context,
    create_llm,
    default_arg_parser,
    ensure_results_dir,
    make_easyagent,
    result_to_dict,
    summarize_results,
    trace_summary,
    write_json,
    now_ms,
)
from core.Exception import ToolConfirmationRequired, ToolInterruption
from easyagent import BasicAgent, Config
from easyagent.permissions import PermissionContext, PermissionMode


def build_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for i in range(6):
        if i % 2 == 0:
            tool = "MockBash"
            params = {"command": f"git status --short # recovery-{i}"}
            category = "shell_confirmation_resume"
        else:
            tool = "MockFileWrite"
            params = {"path": f"notes/recovery_{i}.txt", "content": "approved content"}
            category = "file_write_confirmation_resume"
        task = (
            f"Use benchmark tool `{tool}` with exactly these JSON arguments: {params}. "
            "This is a recovery benchmark. If the tool requires confirmation, stop for confirmation; "
            "after confirmation, continue from the tool result and summarize the result."
        )
        cases.append(
            BenchCase(
                case_id=f"recovery_{i}",
                category=category,
                expected="resumed",
                task=task,
                metadata={"tool": tool, "params": params},
            )
        )
    return cases


def _run_until_interrupt(case: BenchCase, *, permission_context: PermissionContext):
    registry = build_mock_registry()
    agent = make_easyagent(
        registry=registry,
        permission_context=permission_context,
        config=Config(tool_schema_mode="deferred", interrupt_on_confirmation=True),
        system_prompt="You are running a recovery benchmark. Use the requested benchmark tool exactly.",
    )
    interruption = None
    error = None
    output = None
    try:
        output = agent.invoke(case.task, max_iter=6, temperature=0)
    except (ToolConfirmationRequired, ToolInterruption) as exc:
        interruption = exc.to_payload()
    except Exception as exc:
        error = str(exc)
    return agent, registry, output, interruption, error


def run_case(case: BenchCase, *, use_snapshot: bool) -> CaseResult:
    start = now_ms()
    context = build_permission_context()
    agent, registry, initial_output, interruption, initial_error = _run_until_interrupt(
        case,
        permission_context=context,
    )
    initial_trace = trace_summary(agent)

    resumed_output = None
    resume_error = None
    resumed_trace = {}
    observed = "failed"
    duplicate_tool_calls = 0
    pending_restored = False
    session_saved = False

    try:
        if use_snapshot:
            with tempfile.TemporaryDirectory(prefix="easyagent_recovery_") as tmp:
                store_path = str(Path(tmp) / "sessions.sqlite3")
                session_id = f"{case.case_id}_{now_ms()}"
                agent.save_session(session_id, store=store_path, metadata={"case_id": case.case_id})
                session_saved = True
                restored = BasicAgent.load_session(
                    session_id,
                    llm=create_llm(),
                    store=store_path,
                    tool_registry=registry,
                    permission_context=context,
                )
                pending_restored = restored.has_pending_tool_interrupt()
                restored.resolve_last_tool_interrupt(
                    content=f"User approved. Simulated benchmark output for {case.metadata['tool']}: ok.",
                    ephemeral_context={"approved": True, "case_id": case.case_id},
                )
                resumed_output = restored.invoke(
                    "Continue from the approved tool result and provide the final benchmark answer.",
                    max_iter=6,
                    temperature=0,
                    resume_from_history=True,
                )
                resumed_trace = trace_summary(restored)
                observed = "resumed"
        else:
            bypass_context = PermissionContext(mode=PermissionMode.BYPASS)
            restarted_agent = make_easyagent(
                registry=build_mock_registry(),
                permission_context=bypass_context,
                config=Config(tool_schema_mode="deferred", interrupt_on_confirmation=True),
                system_prompt="You are running a restart baseline. Use the requested benchmark tool exactly.",
            )
            resumed_output = restarted_agent.invoke(case.task, max_iter=6, temperature=0)
            resumed_trace = trace_summary(restarted_agent)
            observed = "restarted"
    except Exception as exc:
        resume_error = str(exc)

    before_names = initial_trace.get("tool_names", [])
    after_names = resumed_trace.get("tool_names", [])
    duplicate_tool_calls = sum(1 for name in after_names if name in before_names)
    success = observed == "resumed" if use_snapshot else observed == "restarted"
    if use_snapshot:
        success = success and bool(interruption) and pending_restored

    return CaseResult(
        case_id=f"{'snapshot' if use_snapshot else 'restart'}_{case.case_id}",
        category="recovery_snapshot" if use_snapshot else "recovery_restart_baseline",
        expected="resumed" if use_snapshot else "restarted",
        observed=observed,
        success=success,
        duration_ms=now_ms() - start,
        error=resume_error or initial_error,
        metrics={
            "mode": "snapshot" if use_snapshot else "restart",
            "initial_output": initial_output,
            "initial_interruption": interruption,
            "session_saved": session_saved,
            "pending_restored": pending_restored,
            "resumed_output": resumed_output,
            "initial_trace": initial_trace,
            "resumed_trace": resumed_trace,
            "duplicate_tool_calls": duplicate_tool_calls,
        },
    )


def main() -> None:
    parser = default_arg_parser("E2 real LLM recovery and duplicate side-effect benchmark")
    parser.add_argument("--mode", choices=["snapshot", "restart", "both"], default="snapshot")
    args = parser.parse_args()
    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / f"e2_recovery_{args.mode}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    modes = [args.mode] if args.mode != "both" else ["snapshot", "restart"]
    for case in cases:
        for mode in modes:
            for _ in range(max(args.repeat, 1)):
                result = run_case(case, use_snapshot=mode == "snapshot")
                results.append(result)
                append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary["experiment"] = "e2_recovery"
    summary["mode"] = args.mode
    summary["duplicate_tool_calls_mean"] = (
        sum(r.metrics["duplicate_tool_calls"] for r in results) / len(results) if results else 0
    )
    out_path = Path(args.out) if args.out else out_dir / f"e2_recovery_{args.mode}_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
