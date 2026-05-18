from __future__ import annotations

import argparse
from pathlib import Path

from common import append_jsonl, ensure_results_dir, result_to_dict, summarize_results, write_json
from paper_adapters import AdapterConfig, get_adapter
from paper_cases import select_cases


def run_numbered_experiment(
    *,
    experiment: str,
    description: str,
    default_frameworks: str = "easyagent,langgraph,llamaindex",
    default_repeat: int = 3,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--frameworks", default=default_frameworks)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=default_repeat)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tool-schema-mode", choices=["full", "deferred"], default="deferred")
    parser.add_argument("--permission-mode", choices=["default", "accept_edits", "dont_ask", "bypass"], default="bypass")
    parser.add_argument("--baseline-policy", choices=["native", "guarded"], default="native")
    parser.add_argument("--out", default="")
    parser.add_argument("--jsonl", default="")
    args = parser.parse_args()

    config = AdapterConfig(
        max_iter=args.max_iter,
        temperature=args.temperature,
        timeout_s=args.timeout,
        tool_schema_mode=args.tool_schema_mode,
        permission_mode=args.permission_mode,
        baseline_policy=args.baseline_policy,
    )
    frameworks = [item.strip() for item in args.frameworks.split(",") if item.strip()]
    adapters = [get_adapter(name) for name in frameworks]
    cases = select_cases(experiment, limit=args.limit)

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / f"paper_{experiment}.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results = []
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
            "suite": "paper_numbered_experiment",
            "experiment": experiment,
            "frameworks": frameworks,
            "repeat": args.repeat,
            "limit": args.limit,
            "jsonl": str(jsonl_path),
            "by_framework": _summarize_by_framework(results),
        }
    )
    out_path = Path(args.out) if args.out else out_dir / f"paper_{experiment}_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


def _summarize_by_framework(results) -> dict[str, dict]:
    buckets: dict[str, list] = {}
    for result in results:
        framework = result.metrics.get("framework", "unknown")
        buckets.setdefault(framework, []).append(result)
    return {name: summarize_results(items) for name, items in buckets.items()}
