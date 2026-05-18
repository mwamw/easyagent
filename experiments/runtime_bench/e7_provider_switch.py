from __future__ import annotations

import os
from pathlib import Path

from common import (
    CaseResult,
    append_jsonl,
    build_mock_registry,
    build_permission_context,
    default_arg_parser,
    ensure_results_dir,
    load_env,
    result_to_dict,
    summarize_results,
    trace_summary,
    write_json,
    now_ms,
)
from easyagent import BasicAgent, Config, EasyLLM


def provider_env(provider: str) -> dict[str, str | None]:
    prefix = provider.upper().replace("-", "_")
    return {
        "provider": provider,
        "model": os.getenv(f"EA_{prefix}_MODEL") or os.getenv("EA_MODEL") or os.getenv("LLM_MODEL_ID"),
        "base_url": os.getenv(f"EA_{prefix}_BASE_URL") or os.getenv("EA_BASE_URL") or os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv(f"EA_{prefix}_API_KEY") or os.getenv("EA_API_KEY") or os.getenv("LLM_API_KEY") or "test",
    }


def run_provider(provider: str, *, invoke: bool) -> CaseResult:
    start = now_ms()
    error = None
    observed = "initialized"
    try:
        cfg = provider_env(provider)
        llm = EasyLLM(**cfg, temperature=0, timeout=int(os.getenv("EA_TIMEOUT", "120")))
        registry = build_mock_registry()
        agent = BasicAgent(
            name=f"provider_{provider}",
            llm=llm,
            enable_tool=True,
            tool_registry=registry,
            config=Config(tool_schema_mode="deferred"),
            permission_context=build_permission_context(),
        )
        tool_payload = agent.get_provider_tools()
        output = None
        if invoke:
            output = agent.invoke(
                "Use `MockFileRead` with JSON arguments {'path': 'README.md'}, then answer briefly.",
                max_iter=5,
                temperature=0,
            )
            observed = "invoked"
        return CaseResult(
            case_id=f"provider_{provider}",
            category="provider_switch",
            expected="initialized" if not invoke else "invoked",
            observed=observed,
            success=True,
            duration_ms=now_ms() - start,
            metrics={
                "provider": provider,
                "model": cfg.get("model"),
                "tool_payload_count": len(tool_payload or []),
                "invoke": invoke,
                "output": output,
                "trace": trace_summary(agent),
            },
        )
    except Exception as exc:
        error = str(exc)
        return CaseResult(
            case_id=f"provider_{provider}",
            category="provider_switch",
            expected="initialized" if not invoke else "invoked",
            observed="error",
            success=False,
            duration_ms=now_ms() - start,
            error=error,
            metrics={"provider": provider, "invoke": invoke},
        )


def main() -> None:
    parser = default_arg_parser("E7 provider switch benchmark")
    parser.add_argument("--providers", default="", help="Comma-separated providers. Defaults to EA_PROVIDER_MATRIX or openai,anthropic_native,google_native.")
    parser.add_argument("--no-invoke", action="store_true", help="Only initialize providers; skip real LLM calls.")
    args = parser.parse_args()
    load_env()
    raw = args.providers or os.getenv("EA_PROVIDER_MATRIX") or "openai,anthropic_native,google_native"
    providers = [item.strip() for item in raw.split(",") if item.strip()]
    if args.limit:
        providers = providers[: args.limit]

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "e7_provider_switch.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results: list[CaseResult] = []
    for provider in providers:
        for _ in range(max(args.repeat, 1)):
            result = run_provider(provider, invoke=not args.no_invoke)
            results.append(result)
            append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary["experiment"] = "e7_provider_switch"
    summary["invoke"] = not args.no_invoke
    out_path = Path(args.out) if args.out else out_dir / "e7_provider_switch_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
