from __future__ import annotations

from pathlib import Path

from common import (
    CaseResult,
    append_jsonl,
    build_mock_registry,
    default_arg_parser,
    ensure_results_dir,
    make_easyagent,
    result_to_dict,
    summarize_results,
    trace_summary,
    write_json,
    now_ms,
)
from context.token.counter import TokenCounter
from easyagent import Config


def count_payload_tokens(payload) -> int:
    return TokenCounter().count(payload)


def _run_agent_pick(tool_count: int, *, mode: str, target_index: int, provider: str) -> CaseResult:
    registry = build_mock_registry(generated_tools=tool_count)
    start = now_ms()
    payload = registry.export_tools(provider, mode=mode)
    schema_tokens = count_payload_tokens(payload)
    target = f"GeneratedTool{target_index}"
    agent = make_easyagent(
        registry=registry,
        config=Config(tool_schema_mode=mode),
        system_prompt=(
            "You are running a tool-schema benchmark. Use the exact generated tool requested by the user. "
            "If only a tool inventory is visible, first obtain the missing schema using the available schema mechanism."
        ),
    )
    output = None
    error = None
    try:
        output = agent.invoke(
            f"Use `{target}` with value `needle-{tool_count}-{mode}`. "
            f"The final answer must include the literal marker `generated:{target_index}`.",
            max_iter=8,
            temperature=0,
        )
    except Exception as exc:
        error = str(exc)

    trace = trace_summary(agent)
    called = target in trace["tool_names"]
    observed = "selected_target_tool" if called else "missed_target_tool"
    success = called and not error
    return CaseResult(
        case_id=f"tools_{tool_count}_{mode}_agent",
        category=mode,
        expected="selected_target_tool",
        observed=observed,
        success=success,
        duration_ms=now_ms() - start,
        error=error,
        metrics={
            "provider": provider,
            "tool_count": tool_count,
            "mode": mode,
            "target_tool": target,
            "schema_tokens_initial": schema_tokens,
            "exported_tool_count_initial": len(payload or []),
            "output": output,
            "trace": trace,
        },
    )


def run_size(tool_count: int, *, provider: str, expand_count: int, invoke_agent: bool = True, target_index: int = 7) -> list[CaseResult]:
    results: list[CaseResult] = []
    registry = build_mock_registry(generated_tools=tool_count)

    for mode in ["full", "deferred"]:
        start = now_ms()
        if mode == "deferred" and expand_count > 0:
            names = [f"GeneratedTool{i}" for i in range(min(expand_count, tool_count))]
            registry.expand_deferred_tools(names)
        payload = registry.export_tools(provider, mode=mode)
        tokens = count_payload_tokens(payload)
        tool_names = registry.get_deferred_expanded_tool_names() if mode == "deferred" else []
        results.append(
            CaseResult(
                case_id=f"tools_{tool_count}_{mode}",
                category=mode,
                expected="schema_exported",
                observed="schema_exported",
                success=True,
                duration_ms=now_ms() - start,
                metrics={
                    "provider": provider,
                    "tool_count": tool_count,
                    "mode": mode,
                    "expand_count": expand_count if mode == "deferred" else 0,
                    "schema_tokens": tokens,
                    "exported_tool_count": len(payload or []),
                    "expanded_tools": tool_names,
                },
            )
        )
        registry.clear_deferred_tool_expansions()
    if invoke_agent and tool_count:
        target = max(0, min(target_index, tool_count - 1))
        for mode in ["full", "deferred"]:
            results.append(_run_agent_pick(tool_count, mode=mode, target_index=target, provider=provider))
    return results


def main() -> None:
    parser = default_arg_parser("E3 deferred tool schema benchmark")
    parser.add_argument("--provider", default="openai", help="Provider schema adapter name.")
    parser.add_argument("--sizes", default="10,30,60,100,150", help="Comma-separated tool counts.")
    parser.add_argument("--expand-count", type=int, default=3, help="Number of deferred tools to expand.")
    parser.add_argument("--target-index", type=int, default=7, help="Generated tool index the real agent must select.")
    parser.add_argument("--schema-only", action="store_true", help="Only export schemas; skip real LLM agent calls.")
    args = parser.parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    if args.limit:
        sizes = sizes[: args.limit]

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "e3_deferred_schema.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    for size in sizes:
        for result in run_size(
            size,
            provider=args.provider,
            expand_count=args.expand_count,
            invoke_agent=not args.schema_only,
            target_index=args.target_index,
        ):
            results.append(result)
            append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    pairs: dict[int, dict[str, int]] = {}
    for result in results:
        if "schema_tokens" not in result.metrics:
            continue
        pairs.setdefault(result.metrics["tool_count"], {})[result.category] = result.metrics["schema_tokens"]
    summary["token_reduction_by_size"] = {
        str(size): {
            "full": values.get("full", 0),
            "deferred": values.get("deferred", 0),
            "reduction_ratio": (
                1.0 - values.get("deferred", 0) / values["full"]
                if values.get("full")
                else 0.0
            ),
        }
        for size, values in pairs.items()
    }
    summary["experiment"] = "e3_deferred_schema"
    out_path = Path(args.out) if args.out else out_dir / "e3_deferred_schema_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
