from __future__ import annotations

from pathlib import Path

from common import (
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
from easyagent.permissions import PermissionContext, PermissionMode
from e1_permission_safety import build_cases as build_permission_cases
from e1_permission_safety import run_case as run_permission_case
from e2_recovery import build_cases as build_recovery_cases
from e2_recovery import run_case as run_recovery_case
from e3_deferred_schema import run_size


def permission_ablation(limit: int) -> list[CaseResult]:
    cases = build_permission_cases()
    if limit:
        cases = cases[:limit]
    results: list[CaseResult] = []
    for case in cases:
        full = run_permission_case(case)
        full.case_id = f"full_{case.case_id}"
        full.category = "permission_full"
        results.append(full)

        start = now_ms()
        outcome = run_agent_task(
            case.task,
            registry=build_mock_registry(),
            permission_context=PermissionContext(mode=PermissionMode.BYPASS),
            max_iter=6,
            system_prompt="You are running a permission ablation. Use the requested benchmark tool exactly.",
        )
        observed = outcome["observed"]
        false_allow = case.expected in {"deny", "ask"} and observed == "allow"
        results.append(
            CaseResult(
                case_id=f"wo_permission_{case.case_id}",
                category="permission_wo_permission",
                expected="allow",
                observed=observed,
                success=observed == "allow",
                duration_ms=now_ms() - start,
                error=outcome["error"],
                metrics={
                    "ablation": "wo_permission",
                    "original_expected": case.expected,
                    "false_allow": false_allow,
                    "interruption": outcome["interruption"],
                    "trace": outcome["trace"],
                },
            )
        )
    return results


def recovery_ablation(limit: int) -> list[CaseResult]:
    cases = build_recovery_cases()
    if limit:
        cases = cases[:limit]
    results: list[CaseResult] = []
    for case in cases:
        full = run_recovery_case(case, use_snapshot=True)
        full.category = "recovery_full"
        restart = run_recovery_case(case, use_snapshot=False)
        restart.category = "recovery_wo_snapshot"
        results.extend([full, restart])
    return results


def schema_ablation() -> list[CaseResult]:
    results: list[CaseResult] = []
    for size in [30, 60]:
        for result in run_size(size, provider="openai", expand_count=3, invoke_agent=True):
            result.category = f"schema_{result.category}"
            results.append(result)
    return results


def main() -> None:
    parser = default_arg_parser("E5 EasyAgent ablation benchmark")
    args = parser.parse_args()
    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "e5_ablation.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    results.extend(permission_ablation(args.limit))
    results.extend(recovery_ablation(args.limit))
    results.extend(schema_ablation())

    for result in results:
        append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary["experiment"] = "e5_ablation"
    summary["notes"] = {
        "permission_full": "Expected allow/ask/deny behavior from benchmark rules.",
        "permission_wo_permission": "Bypass mode; use false-allow rate from detailed JSONL.",
        "recovery_wo_snapshot": "Restart-like recovery to expose duplicate side effects.",
        "schema_full_vs_deferred": "Token cost change from deferred schema.",
    }
    out_path = Path(args.out) if args.out else out_dir / "e5_ablation_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
