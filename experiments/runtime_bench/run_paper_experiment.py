from __future__ import annotations

import argparse
from pathlib import Path

from common import append_jsonl, ensure_results_dir, result_to_dict, summarize_results, write_json
from paper_adapters import AdapterConfig, get_adapter
from paper_cases import all_experiments, select_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-style cross-framework EasyAgent experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=["all", *all_experiments()],
        default="permission",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--frameworks",
        default="easyagent,langgraph,llamaindex",
        help="Comma-separated frameworks: easyagent, langgraph, langchain, llamaindex.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit cases per experiment. 0 means all.")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat count per case.")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tool-schema-mode", choices=["full", "deferred"], default="deferred")
    parser.add_argument("--permission-mode", choices=["default", "accept_edits", "dont_ask", "bypass"], default="bypass")
    parser.add_argument(
        "--baseline-policy",
        choices=["native", "guarded"],
        default="native",
        help="native measures baseline out-of-box behavior; guarded adds an external policy wrapper.",
    )
    parser.add_argument("--out", default="", help="Optional summary JSON path.")
    parser.add_argument("--jsonl", default="", help="Optional per-run JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = all_experiments() if args.experiment == "all" else [args.experiment]
    framework_names = [item.strip() for item in args.frameworks.split(",") if item.strip()]
    adapters = [get_adapter(name) for name in framework_names]
    config = AdapterConfig(
        max_iter=args.max_iter,
        temperature=args.temperature,
        timeout_s=args.timeout,
        tool_schema_mode=args.tool_schema_mode,
        permission_mode=args.permission_mode,
        baseline_policy=args.baseline_policy,
    )

    out_dir = ensure_results_dir()
    suffix = args.experiment if args.experiment != "all" else "all"
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / f"paper_{suffix}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results = []
    for experiment in experiments:
        cases = select_cases(experiment, limit=args.limit)
        for case in cases:
            for adapter in adapters:
                for repeat_idx in range(max(args.repeat, 1)):
                    result = adapter.run(experiment, case, config)
                    result.metrics = {
                        "framework": adapter.name,
                        "experiment": experiment,
                        "repeat_idx": repeat_idx,
                        **dict(result.metrics or {}),
                    }
                    results.append(result)
                    append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary.update(
        {
            "suite": "paper_cross_framework",
            "experiment": args.experiment,
            "frameworks": framework_names,
            "repeat": args.repeat,
            "limit": args.limit,
            "jsonl": str(jsonl_path),
        }
    )
    summary["by_framework"] = summarize_by_framework(results)
    out_path = Path(args.out) if args.out else out_dir / f"paper_{suffix}_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


def summarize_by_framework(results) -> dict[str, dict]:
    buckets: dict[str, list] = {}
    for result in results:
        framework = result.metrics.get("framework", "unknown")
        buckets.setdefault(framework, []).append(result)
    return {name: summarize_results(items) for name, items in buckets.items()}


if __name__ == "__main__":
    main()
