from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_SCRIPTS = [
    "p1_multimodel_adaptation.py",
    "p2_permission_safety_compare.py",
    "p3_code_agent_compare.py",
    "p4_recovery_fault_injection.py",
    "p5_deferred_schema_efficiency.py",
    "p6_multi_agent_collaboration.py",
    "p7_observability_debug.py",
    "p8_ablation_study.py",
    "p9_engineering_complexity.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all paper experiment scripts sequentially.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--frameworks", default="easyagent,langgraph,llamaindex")
    parser.add_argument("--only", default="", help="Comma-separated script stem filters, for example p2,p5.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    selected = EXPERIMENT_SCRIPTS
    if args.only:
        filters = tuple(item.strip() for item in args.only.split(",") if item.strip())
        selected = [name for name in EXPERIMENT_SCRIPTS if name.startswith(filters)]

    for script in selected:
        cmd = [
            sys.executable,
            str(root / script),
            "--limit",
            str(args.limit),
            "--repeat",
            str(args.repeat),
        ]
        if script != "p8_ablation_study.py":
            cmd.extend(["--frameworks", args.frameworks])
        print("Running:", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=str(root.parents[1]), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
