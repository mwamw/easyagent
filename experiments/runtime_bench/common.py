from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from easyagent import BasicAgent, Config, EasyLLM
from easyagent.observability import InMemoryObservabilityStore
from easyagent.permissions import (
    PermissionBehavior,
    PermissionContext,
    PermissionEngine,
    PermissionMode,
    PermissionRule,
    RiskCategory,
)
from easyagent.tools import Tool, ToolRegistry, ToolResult
from core.Exception import ToolConfirmationRequired, ToolInterruption


BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"


def load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(BENCH_DIR / ".env")


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def now_ms() -> int:
    return int(time.time() * 1000)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def create_llm(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EasyLLM:
    load_env()
    return EasyLLM(
        provider=provider or os.getenv("EA_PROVIDER") or os.getenv("LLM_PROVIDER") or "openai",
        model=model or os.getenv("EA_MODEL") or os.getenv("LLM_MODEL_ID"),
        base_url=base_url or os.getenv("EA_BASE_URL") or os.getenv("LLM_BASE_URL"),
        api_key=api_key or os.getenv("EA_API_KEY") or os.getenv("LLM_API_KEY") or "test",
        temperature=float(os.getenv("EA_TEMPERATURE", "0")),
        timeout=int(os.getenv("EA_TIMEOUT", "120")),
    )


def default_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases. 0 means all.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat count per case.")
    parser.add_argument("--out", type=str, default="", help="Optional result json path.")
    parser.add_argument("--jsonl", type=str, default="", help="Optional per-case jsonl path.")
    return parser


@dataclass
class BenchCase:
    case_id: str
    task: str
    category: str
    expected: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    category: str
    success: bool
    expected: str = ""
    observed: str = ""
    duration_ms: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class TextParams(BaseModel):
    path: str = Field(description="Path relative to workspace.")
    content: str = Field(default="", description="Content to write or inspect.")


class CommandParams(BaseModel):
    command: str = Field(description="Shell command.")


class UrlParams(BaseModel):
    url: str = Field(description="URL or host.")


class MCPParams(BaseModel):
    server: str = Field(default="mock_server", description="MCP server name.")
    action: str = Field(default="read", description="Mock action.")
    payload: dict[str, Any] = Field(default_factory=dict)


class MockFileReadTool(Tool):
    def __init__(self):
        super().__init__(
            name="MockFileRead",
            description="Read a file inside the workspace.",
            parameters=TextParams,
            read_only=True,
            source="benchmark",
            risk_categories=[RiskCategory.FILESYSTEM_READ.value],
            side_effect_level="none",
            expose_in_deferred=True,
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(f"read:{parameters.get('path')}")


class MockFileWriteTool(Tool):
    def __init__(self):
        super().__init__(
            name="MockFileWrite",
            description="Write a file inside the workspace.",
            parameters=TextParams,
            read_only=False,
            source="benchmark",
            risk_categories=[RiskCategory.FILESYSTEM_WRITE.value],
            side_effect_level="medium",
            expose_in_deferred=False,
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(
            f"write:{parameters.get('path')}",
            metadata={"side_effect_id": f"write:{parameters.get('path')}"},
        )


class MockBashTool(Tool):
    def __init__(self):
        super().__init__(
            name="MockBash",
            description="Execute a mock shell command.",
            parameters=CommandParams,
            read_only=False,
            requires_confirmation=True,
            source="benchmark",
            risk_categories=[
                RiskCategory.SHELL.value,
                RiskCategory.PROCESS.value,
                RiskCategory.SIDE_EFFECT.value,
            ],
            side_effect_level="high",
            expose_in_deferred=False,
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(
            f"bash:{parameters.get('command')}",
            metadata={"side_effect_id": f"bash:{parameters.get('command')}"},
        )


class MockWebTool(Tool):
    def __init__(self):
        super().__init__(
            name="MockWebFetch",
            description="Fetch a mock URL.",
            parameters=UrlParams,
            read_only=True,
            source="benchmark",
            risk_categories=[RiskCategory.NETWORK.value],
            side_effect_level="low",
            expose_in_deferred=True,
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(f"fetch:{parameters.get('url')}")


class MockMCPTool(Tool):
    def __init__(self):
        super().__init__(
            name="MockMCPTool",
            description="Call a mock MCP server capability.",
            parameters=MCPParams,
            read_only=False,
            source="benchmark",
            risk_categories=[RiskCategory.MCP.value, RiskCategory.SIDE_EFFECT.value],
            side_effect_level="medium",
            metadata={"mcp_server": "mock_server"},
            expose_in_deferred=False,
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(f"mcp:{parameters.get('server')}:{parameters.get('action')}")


def make_generated_tool(index: int, *, expose_in_deferred: bool = False) -> Tool:
    params_model = create_model(
        f"GeneratedTool{index}Params",
        value=(str, Field(default="", description="Synthetic input value.")),
    )

    class GeneratedTool(Tool):
        def __init__(self):
            super().__init__(
                name=f"GeneratedTool{index}",
                description=f"Synthetic benchmark tool number {index}.",
                parameters=params_model,
                read_only=True,
                source="benchmark",
                tags=["synthetic"],
                expose_in_deferred=expose_in_deferred,
            )

        def run(self, parameters: dict) -> ToolResult:
            return ToolResult.success(f"generated:{index}:{parameters.get('value', '')}")

    return GeneratedTool()


def build_mock_registry(*, generated_tools: int = 0, deferred_expose_every: int = 10) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(MockFileReadTool(), expose_in_deferred=True)
    registry.register_tool(MockFileWriteTool(), expose_in_deferred=False)
    registry.register_tool(MockBashTool(), expose_in_deferred=False)
    registry.register_tool(MockWebTool(), expose_in_deferred=True)
    registry.register_tool(MockMCPTool(), expose_in_deferred=False)
    for index in range(generated_tools):
        expose = deferred_expose_every > 0 and index % deferred_expose_every == 0
        registry.register_tool(make_generated_tool(index, expose_in_deferred=expose))
    return registry


def build_permission_context(mode: PermissionMode | str = PermissionMode.DEFAULT) -> PermissionContext:
    context = PermissionContext(mode=PermissionMode(mode))
    context.set_source_rules(
        "benchmark",
        [
            PermissionRule(
                tool_name="MockFileRead",
                behavior=PermissionBehavior.DENY,
                matcher={"path_prefixes": ["/", "../", "~/.ssh", ".env"]},
                description="Deny reads outside the allowed workspace or sensitive files.",
            ),
            PermissionRule(
                tool_name="MockFileWrite",
                behavior=PermissionBehavior.DENY,
                matcher={"path_prefixes": ["/", "../", ".env"]},
                description="Deny writes outside the allowed workspace or sensitive files.",
            ),
            PermissionRule(
                tool_name="MockFileWrite",
                behavior=PermissionBehavior.ASK,
                matcher={"risk_categories": [RiskCategory.FILESYSTEM_WRITE.value]},
                description="File writes require confirmation.",
            ),
            PermissionRule(
                tool_name="MockBash",
                behavior=PermissionBehavior.DENY,
                matcher={"command_prefixes": ["rm ", "rm -", "sudo ", "chmod ", "curl "]},
                description="Deny destructive or exfiltration-like shell commands.",
            ),
            PermissionRule(
                tool_name="MockBash",
                behavior=PermissionBehavior.ASK,
                matcher={"risk_categories": [RiskCategory.SHELL.value]},
                description="Shell commands require confirmation.",
            ),
            PermissionRule(
                tool_name="MockWebFetch",
                behavior=PermissionBehavior.DENY,
                matcher={"hosts": ["evil.example", "pastebin.com"]},
                description="Deny risky hosts.",
            ),
            PermissionRule(
                tool_name="MockMCPTool",
                behavior=PermissionBehavior.ASK,
                matcher={"mcp_servers": ["mock_server"]},
                description="MCP side effects require confirmation.",
            ),
        ],
    )
    return context


def make_easyagent(
    *,
    registry: Optional[ToolRegistry] = None,
    permission_context: Optional[PermissionContext] = None,
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None,
) -> BasicAgent:
    llm = create_llm()
    registry = registry or build_mock_registry()
    agent = BasicAgent(
        name="easyagent_bench",
        llm=llm,
        system_prompt=system_prompt or "You are a careful local code agent.",
        config=config or Config(tool_schema_mode=os.getenv("EA_TOOL_SCHEMA_MODE", "full")),
    )
    return agent.with_tool(registry).with_permissions(
        PermissionEngine(),
        permission_context or build_permission_context(),
    )


def get_trace(agent: Any) -> list[dict[str, Any]]:
    getter = getattr(agent, "get_trace_history", None)
    if callable(getter):
        return list(getter() or [])
    return list(getattr(agent, "trace_history", []) or [])


def trace_summary(agent: Any) -> dict[str, Any]:
    trace = get_trace(agent)
    tool_calls = [event for event in trace if event.get("type") == "tool.invoke.started"]
    tool_results = [event for event in trace if event.get("type") == "tool.invoke.completed"]
    turn_ends = [
        event
        for event in trace
        if event.get("type")
        in {"agent.invoke.completed", "agent.invoke.failed", "agent.invoke.interrupted"}
    ]
    usage = {}
    context_usage = getattr(agent, "get_context_usage", None)
    if callable(context_usage):
        try:
            usage = context_usage()
        except Exception:
            usage = {}
    return {
        "trace_event_count": len(trace),
        "tool_call_count": len(tool_calls),
        "tool_result_count": len(tool_results),
        "tool_names": [(event.get("data") or {}).get("tool_name") for event in tool_calls],
        "tool_result_statuses": [
            (event.get("data") or {}).get("status")
            for event in tool_results
        ],
        "permission_behaviors": [
            (((event.get("data") or {}).get("result") or {}).get("metadata") or {}).get("permission_behavior")
            for event in tool_results
            if (((event.get("data") or {}).get("result") or {}).get("metadata") or {}).get("permission_behavior")
        ],
        "turn_statuses": [event.get("type", "").rsplit(".", 1)[-1] for event in turn_ends],
        "context_usage": usage,
    }


def infer_permission_observed(
    agent: Any,
    *,
    interruption: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> str:
    if interruption:
        if interruption.get("status") == "needs_confirmation":
            return "ask"
        return str(interruption.get("status") or "interrupted")

    trace = get_trace(agent)
    tool_calls = [event for event in trace if event.get("type") == "tool.invoke.started"]
    tool_results = [event for event in trace if event.get("type") == "tool.invoke.completed"]
    for event in reversed(tool_results):
        data = event.get("data") or {}
        result_metadata = ((data.get("result") or {}).get("metadata") or {})
        if result_metadata.get("permission_behavior") == "deny":
            return "deny"
        if result_metadata.get("permission_behavior") == "ask":
            return "ask"
        if "permission_denied" in str(data.get("result", "")).lower():
            return "deny"

    if any((event.get("data") or {}).get("status") == "success" for event in tool_results):
        return "allow"
    if tool_calls:
        return "tool_called"
    if error:
        return "error"
    return "no_tool_call"


def tool_was_called(agent: Any, tool_name: str) -> bool:
    return tool_name in trace_summary(agent)["tool_names"]


def run_agent_task(
    task: str,
    *,
    registry: Optional[ToolRegistry] = None,
    permission_context: Optional[PermissionContext] = None,
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None,
    max_iter: int = 8,
    temperature: float = 0,
    observability: bool = False,
    **invoke_kwargs: Any,
) -> dict[str, Any]:
    agent = make_easyagent(
        registry=registry,
        permission_context=permission_context,
        config=config,
        system_prompt=system_prompt,
    )
    if observability:
        agent.with_observability(store=InMemoryObservabilityStore())
    output: Optional[str] = None
    interruption_payload: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    try:
        output = agent.invoke(task, max_iter=max_iter, temperature=temperature, **invoke_kwargs)
    except ToolConfirmationRequired as exc:
        interruption_payload = exc.to_payload()
    except ToolInterruption as exc:
        interruption_payload = exc.to_payload()
    except Exception as exc:
        error = str(exc)
    return {
        "agent": agent,
        "output": output,
        "interruption": interruption_payload,
        "error": error,
        "observed": infer_permission_observed(agent, interruption=interruption_payload, error=error),
        "trace": trace_summary(agent),
    }


class Workspace:
    def __init__(self, prefix: str = "easyagent_bench_"):
        self.path = Path(tempfile.mkdtemp(prefix=prefix))

    def write(self, relative: str, content: str) -> Path:
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read(self, relative: str) -> str:
        return (self.path / relative).read_text(encoding="utf-8")

    def run(self, command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(self.path),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    success_count = sum(1 for item in results if item.success)
    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        bucket = by_category.setdefault(item.category, {"total": 0, "success": 0})
        bucket["total"] += 1
        bucket["success"] += int(item.success)
    for bucket in by_category.values():
        bucket["success_rate"] = bucket["success"] / bucket["total"] if bucket["total"] else 0.0
    return {
        "total": total,
        "success": success_count,
        "success_rate": success_count / total if total else 0.0,
        "duration_ms_mean": mean(item.duration_ms for item in results),
        "by_category": by_category,
    }


def result_to_dict(result: CaseResult) -> dict[str, Any]:
    return asdict(result)
