from __future__ import annotations

from paper_script_common import run_numbered_experiment


if __name__ == "__main__":
    run_numbered_experiment(
        experiment="multimodel",
        description="P1 multi-model/provider adaptation experiment",
        default_frameworks="easyagent",
    )
