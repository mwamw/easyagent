from __future__ import annotations

from paper_script_common import run_numbered_experiment


if __name__ == "__main__":
    run_numbered_experiment(
        experiment="complexity",
        description="P9 engineering complexity and glue-code cost experiment",
        default_frameworks="easyagent,langgraph,langchain,llamaindex",
        default_repeat=1,
    )
