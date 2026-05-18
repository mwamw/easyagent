from __future__ import annotations

import os
from typing import Iterable

from common import BenchCase
from e1_permission_safety import build_cases as build_permission_cases
from e2_recovery import build_cases as build_recovery_cases
from e4_code_agent_tasks import build_cases as build_code_cases
from e6_observability_debug import build_cases as build_observability_cases


def select_cases(experiment: str, *, limit: int = 0) -> list[BenchCase]:
    if experiment == "permission":
        cases = build_permission_cases()
    elif experiment == "recovery":
        cases = build_recovery_cases()
    elif experiment == "code":
        cases = build_code_cases()
    elif experiment == "schema":
        cases = build_schema_cases()
    elif experiment == "multimodel":
        cases = build_multimodel_cases()
    elif experiment == "observability":
        cases = build_observability_cases()
    elif experiment == "multi_agent":
        cases = build_multi_agent_cases()
    elif experiment == "complexity":
        cases = build_complexity_cases()
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return cases[:limit] if limit else cases


def build_schema_cases(sizes: Iterable[int] | None = None) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for size in sizes or [10, 30, 60, 100]:
        target_index = min(7, size - 1)
        cases.append(
            BenchCase(
                case_id=f"schema_tools_{size}",
                category="deferred_schema",
                expected="selected_target_tool",
                task=(
                    f"Use `GeneratedTool{target_index}` with value `paper-bench-{size}`. "
                    f"The final answer must include `generated:{target_index}`."
                ),
                metadata={"tool_count": size, "target_index": target_index},
            )
        )
    return cases


def all_experiments() -> list[str]:
    return [
        "multimodel",
        "permission",
        "code",
        "recovery",
        "schema",
        "multi_agent",
        "observability",
        "complexity",
    ]


def build_multimodel_cases() -> list[BenchCase]:
    raw = os.getenv("EA_PROVIDER_MATRIX") or "openai,anthropic_native,google_native"
    providers = [item.strip() for item in raw.split(",") if item.strip()]
    return [
        BenchCase(
            case_id=f"provider_{provider}",
            category="provider_switch",
            expected="invoked",
            task="Use MockFileRead to read README.md and answer briefly.",
            metadata={"provider": provider},
        )
        for provider in providers
    ]


def build_multi_agent_cases() -> list[BenchCase]:
    return [
        BenchCase(
            case_id="multi_agent_foreground",
            category="multi_agent_runtime",
            expected="completed",
            task="Run one foreground subagent that reads module_a.py and reports a concise finding.",
            metadata={"background": False},
        ),
        BenchCase(
            case_id="multi_agent_background",
            category="multi_agent_runtime",
            expected="completed",
            task="Run one background subagent that reads module_a.py and reports a concise finding.",
            metadata={"background": True},
        ),
    ]


def build_complexity_cases() -> list[BenchCase]:
    return [
        BenchCase(
            case_id="runtime_feature_surface",
            category="engineering_complexity",
            expected="measured",
            task=(
                "Measure the framework code needed to implement a local code agent with permission, "
                "file tools, shell, recovery, trace, and tool registration."
            ),
            metadata={},
        )
    ]
