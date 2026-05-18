# EasyAgent Runtime Benchmarks

This folder contains experiment scaffolding for evaluating EasyAgent as a
permission-aware, recoverable local Code Agent runtime.

Copy `.env.example` to either repo root `.env` or this directory `.env`, then
fill model settings. All experiments in this folder execute a real EasyAgent
LLM loop by default, including tool selection, permission handling, trace
recording, and recovery paths.

## Files

- `e1_permission_safety.py`: real-agent permission allow/ask/deny benchmark.
- `e2_recovery.py`: real-agent interruption recovery and duplicate tool-call benchmark.
- `e3_deferred_schema.py`: deferred tool schema token-cost and tool-selection benchmark.
- `e4_code_agent_tasks.py`: small local code-agent task benchmark.
- `e5_ablation.py`: combined ablation benchmark.
- `e6_observability_debug.py`: observability/debug completeness benchmark.
- `e7_provider_switch.py`: provider switching benchmark with real invocation by default.
- `e8_multi_agent_case.py`: real subagent runtime case study.
- `paper_cases.py`: shared case builders for paper-style cross-framework runs.
- `paper_adapters.py`: EasyAgent/LangGraph/LangChain/LlamaIndex adapter layer.
- `run_paper_experiment.py`: unified cross-framework runner.
- `analyze_paper_results.py`: compact JSONL result summarizer.
- `statistical_analysis.py`: mean/std and bootstrap confidence interval report.
- `p1_multimodel_adaptation.py`: provider/model switching experiment.
- `p2_permission_safety_compare.py`: permission safety comparison.
- `p3_code_agent_compare.py`: local code-agent task comparison.
- `p4_recovery_fault_injection.py`: recovery/failure-injection comparison.
- `p5_deferred_schema_efficiency.py`: deferred schema token-efficiency comparison.
- `p6_multi_agent_collaboration.py`: multi-agent runtime comparison.
- `p7_observability_debug.py`: observability/debugging comparison.
- `p8_ablation_study.py`: EasyAgent ablation study.
- `p9_engineering_complexity.py`: engineering-complexity comparison.
- `run_all_paper_experiments.py`: sequential runner for the numbered scripts.
- `common.py`: shared tools, result schema, env/model helpers.

## Quick Start

Run from the repository root:

```bash
python experiments/runtime_bench/e1_permission_safety.py
python experiments/runtime_bench/e2_recovery.py --mode snapshot
python experiments/runtime_bench/e2_recovery.py --mode restart
python experiments/runtime_bench/e3_deferred_schema.py
python experiments/runtime_bench/e4_code_agent_tasks.py --tool-schema-mode deferred --permission-mode bypass
python experiments/runtime_bench/e5_ablation.py --limit 50
python experiments/runtime_bench/e6_observability_debug.py
python experiments/runtime_bench/e7_provider_switch.py
python experiments/runtime_bench/e8_multi_agent_case.py
```

Paper-style cross-framework entry point:

```bash
python experiments/runtime_bench/run_paper_experiment.py \
  --experiment permission \
  --frameworks easyagent,langgraph,llamaindex \
  --limit 10 \
  --repeat 3

python experiments/runtime_bench/run_paper_experiment.py \
  --experiment schema \
  --frameworks easyagent,langgraph,langchain \
  --limit 2 \
  --repeat 3

python experiments/runtime_bench/analyze_paper_results.py \
  experiments/runtime_bench/results/paper_permission.jsonl

python experiments/runtime_bench/statistical_analysis.py \
  experiments/runtime_bench/results/paper_permission.jsonl
```

Numbered paper experiment scripts:

```bash
python experiments/runtime_bench/p1_multimodel_adaptation.py --limit 2 --repeat 3
python experiments/runtime_bench/p2_permission_safety_compare.py --limit 10 --repeat 3
python experiments/runtime_bench/p3_code_agent_compare.py --limit 3 --repeat 1
python experiments/runtime_bench/p4_recovery_fault_injection.py --limit 3 --repeat 3
python experiments/runtime_bench/p5_deferred_schema_efficiency.py --limit 2 --repeat 3
python experiments/runtime_bench/p6_multi_agent_collaboration.py --repeat 1
python experiments/runtime_bench/p7_observability_debug.py --repeat 3
python experiments/runtime_bench/p8_ablation_study.py --limit 5 --repeat 1
python experiments/runtime_bench/p9_engineering_complexity.py --repeat 1
```

Run all numbered scripts sequentially:

```bash
python experiments/runtime_bench/run_all_paper_experiments.py \
  --frameworks easyagent,langgraph,llamaindex \
  --limit 3 \
  --repeat 3
```

For quick smoke tests, run with small limits:

```bash
python experiments/runtime_bench/e1_permission_safety.py --limit 3
python experiments/runtime_bench/e2_recovery.py --mode snapshot --limit 1
python experiments/runtime_bench/e3_deferred_schema.py --sizes 10 --limit 1
```

Results are written to `experiments/runtime_bench/results/` as per-case JSONL
and summary JSON files.

## Notes

- Because these are real LLM-agent experiments, use `--limit` and `--repeat`
  deliberately to control API cost and variance.
- `e3_deferred_schema.py --schema-only` can be used for token accounting only,
  but the default includes real tool-selection calls.
- `e7_provider_switch.py --no-invoke` can be used to debug provider setup only,
  but the default invokes the model.
- `run_paper_experiment.py` is the paper comparison path. EasyAgent is fully
  bound to the local runtime. LangGraph/LangChain adapters require optional
  packages (`langgraph`, `langchain`, `langchain-core`, `langchain-openai`).
  LlamaIndex is scaffolded and currently reports `experiment_unsupported` until
  its installed version's AgentWorkflow API is bound explicitly.
- Baseline `--baseline-policy native` measures framework out-of-box tool
  behavior. `--baseline-policy guarded` adds an external policy wrapper around
  baseline tools, which is useful for engineering-complexity and fair-tooling
  variants but should be reported separately in the paper.
