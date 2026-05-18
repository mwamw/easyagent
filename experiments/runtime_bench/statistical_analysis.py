from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean/std and bootstrap CI for paper benchmark JSONL.")
    parser.add_argument("jsonl")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    random.seed(args.seed)

    records = []
    with Path(args.jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    groups = defaultdict(list)
    for record in records:
        metrics = record.get("metrics") or {}
        key = (metrics.get("experiment", record.get("category")), metrics.get("framework", "unknown"))
        groups[key].append(record)

    summary = {}
    for (experiment, framework), items in groups.items():
        success = [1.0 if item.get("success") else 0.0 for item in items]
        durations = [float(item.get("duration_ms") or 0) for item in items]
        summary.setdefault(experiment, {})[framework] = {
            "n": len(items),
            "success_rate_mean": mean(success) if success else 0.0,
            "success_rate_std": pstdev(success) if len(success) > 1 else 0.0,
            "success_rate_bootstrap_ci95": bootstrap_ci(success, args.bootstrap),
            "duration_ms_mean": mean(durations) if durations else 0.0,
            "duration_ms_std": pstdev(durations) if len(durations) > 1 else 0.0,
            "duration_ms_bootstrap_ci95": bootstrap_ci(durations, args.bootstrap),
        }

    payload = {"source": args.jsonl, "summary": summary}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def bootstrap_ci(values: list[float], rounds: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    samples = []
    for _ in range(max(rounds, 1)):
        draw = [random.choice(values) for _ in values]
        samples.append(mean(draw))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [lo, hi]


if __name__ == "__main__":
    main()
