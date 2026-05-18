from __future__ import annotations

import argparse
from pathlib import Path

from common import append_jsonl, ensure_results_dir, result_to_dict, summarize_results, write_json
from e5_ablation import permission_ablation, recovery_ablation, schema_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="P8 EasyAgent ablation study")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--parts",
        default="permission,recovery,schema",
        help="Comma-separated ablation parts: permission,recovery,schema.",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--jsonl", default="")
    args = parser.parse_args()

    out_dir = ensure_results_dir()
    jsonl_path = Path(args.jsonl) if args.jsonl else out_dir / "paper_ablation.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    parts = {item.strip() for item in args.parts.split(",") if item.strip()}
    results = []
    for repeat_idx in range(max(args.repeat, 1)):
        batch = []
        if "permission" in parts:
            batch.extend(permission_ablation(args.limit))
        if "recovery" in parts:
            batch.extend(recovery_ablation(args.limit))
        if "schema" in parts:
            batch.extend(schema_ablation())
        for result in batch:
            result.metrics = {
                "framework": "easyagent",
                "experiment": "ablation",
                "repeat_idx": repeat_idx,
                **dict(result.metrics or {}),
            }
            results.append(result)
            append_jsonl(jsonl_path, result_to_dict(result))

    summary = summarize_results(results)
    summary.update(
        {
            "suite": "paper_numbered_experiment",
            "experiment": "ablation",
            "parts": sorted(parts),
            "repeat": args.repeat,
            "limit": args.limit,
            "jsonl": str(jsonl_path),
        }
    )
    out_path = Path(args.out) if args.out else out_dir / "paper_ablation_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
