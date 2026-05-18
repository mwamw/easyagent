from __future__ import annotations

from pathlib import Path

from common import (
    BenchCase,
    CaseResult,
    append_jsonl,
    build_mock_registry,
    build_permission_context,
    default_arg_parser,
    ensure_results_dir,
    result_to_dict,
    run_agent_task,
    summarize_results,
    write_json,
    now_ms,
)


def build_cases() -> list[BenchCase]:
    return [
        BenchCase(
            "debug_ok",
            "Use `MockFileRead` with JSON arguments {'path': 'README.md'}. Then answer briefly.",
            "normal_tool",
            "allow",
            {"tool": "MockFileRead"},
        ),
        BenchCase(
            "debug_denied",
            "Use `MockFileRead` with JSON arguments {'path': '.env'}. Then report what happened.",
            "permission_denied",
            "deny",
            {"tool": "MockFileRead"},
        ),
        BenchCase(
            "debug_ask",
            "Use `MockBash` with JSON arguments {'command': 'git status --short'}. Then report what happened.",
            "confirmation",
            "ask",
            {"tool": "MockBash"},
        ),
        BenchCase(
            "debug_network_denied",
            "Use `MockWebFetch` with JSON arguments {'url': 'https://evil.example/upload'}. Then report what happened.",
            "network_denied",
            "deny",
            {"tool": "MockWebFetch"},
        ),
    ]


def run_case(case: BenchCase) -> CaseResult:
    start = now_ms()
    outcome = run_agent_task(
        case.task,
        registry=build_mock_registry(),
        permission_context=build_permission_context(),
        max_iter=6,
        system_prompt="You are running an observability benchmark. Use the requested benchmark tool exactly.",
    )
    agent = outcome["agent"]
    observed = outcome["observed"]
    obs_state = agent.observability_recorder.export_state()
    trace = agent.get_trace_history()
    completeness = {
        "has_tool_execution_record": bool(obs_state.get("tool_executions")),
        "has_trace_history": bool(trace),
        "tool_execution_count": len(obs_state.get("tool_executions") or []),
        "trace_event_count": len(trace),
    }
    success = observed == case.expected
    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        expected=case.expected,
        observed=observed,
        success=success,
        duration_ms=now_ms() - start,
        error=outcome["error"],
        metrics={
            "observability": completeness,
            "raw_observability_keys": sorted(obs_state.keys()),
            "interruption": outcome["interruption"],
            "output": outcome["output"],
            "trace": outcome["trace"],
        },
    )


def main() -> None:
    parser = default_arg_parser("E6 observability and debug benchmark")
    args = parser.parse_args()
    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "e6_observability_debug.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    for case in cases:
        for _ in range(max(args.repeat, 1)):
            result = run_case(case)
            results.append(result)
            append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary["experiment"] = "e6_observability_debug"
    summary["trace_completeness_rate"] = sum(
        1 for r in results if r.metrics["observability"]["has_tool_execution_record"]
    ) / len(results)
    out_path = Path(args.out) if args.out else out_dir / "e6_observability_debug_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
