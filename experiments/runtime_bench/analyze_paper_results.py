from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper benchmark JSONL results.")
    parser.add_argument("jsonl", help="Path to paper_*.jsonl")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    records = []
    with Path(args.jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    grouped = defaultdict(list)
    for record in records:
        metrics = record.get("metrics") or {}
        key = (metrics.get("experiment", record.get("category")), metrics.get("framework", "unknown"))
        grouped[key].append(record)

    summary = {}
    for (experiment, framework), items in grouped.items():
        durations = [float(item.get("duration_ms") or 0) for item in items]
        successes = [1.0 if item.get("success") else 0.0 for item in items]
        observed_counts = defaultdict(int)
        for item in items:
            observed_counts[str(item.get("observed"))] += 1
        summary.setdefault(experiment, {})[framework] = {
            "n": len(items),
            "success_rate": mean(successes) if successes else 0.0,
            "duration_ms_mean": mean(durations) if durations else 0.0,
            "duration_ms_std": pstdev(durations) if len(durations) > 1 else 0.0,
            "observed_counts": dict(sorted(observed_counts.items())),
        }

    payload = {"source": args.jsonl, "summary": summary}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
