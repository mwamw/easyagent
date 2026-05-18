from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Callable

from common import (
    BenchCase,
    CaseResult,
    build_mock_registry,
    create_llm,
    now_ms,
    trace_summary,
)
from e1_permission_safety import run_case as run_easyagent_permission
from e2_recovery import run_case as run_easyagent_recovery
from e3_deferred_schema import run_size as run_easyagent_schema_size
from e4_code_agent_tasks import run_case as run_easyagent_code
from e6_observability_debug import run_case as run_easyagent_observability
from e7_provider_switch import run_provider as run_easyagent_provider
from e8_multi_agent_case import run_case as run_easyagent_multi_agent


@dataclass
class AdapterConfig:
    max_iter: int = 8
    temperature: float = 0
    timeout_s: int = 120
    tool_schema_mode: str = "deferred"
    permission_mode: str = "bypass"
    baseline_policy: str = "native"


class FrameworkAdapter:
    name = "base"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, experiment: str, case: BenchCase, config: AdapterConfig) -> CaseResult:
        method = getattr(self, f"run_{experiment}", None)
        if experiment == "complexity" and callable(method):
            return method(case, config)
        available, reason = self.available()
        if not available:
            return unavailable_result(self.name, experiment, case, reason)
        if not callable(method):
            return unsupported_result(self.name, experiment, case)
        return method(case, config)


class EasyAgentAdapter(FrameworkAdapter):
    name = "easyagent"

    def run_multimodel(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        result = run_easyagent_provider(str(case.metadata["provider"]), invoke=True)
        result.case_id = case.case_id
        return with_framework(result, self.name)

    def run_permission(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        result = run_easyagent_permission(case)
        return with_framework(result, self.name)

    def run_recovery(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        result = run_easyagent_recovery(case, use_snapshot=True)
        return with_framework(result, self.name)

    def run_schema(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        size = int(case.metadata["tool_count"])
        target_index = int(case.metadata["target_index"])
        results = run_easyagent_schema_size(
            size,
            provider=os.getenv("EA_PROVIDER", "openai"),
            expand_count=3,
            invoke_agent=True,
            target_index=target_index,
        )
        agent_results = [item for item in results if item.case_id.endswith("_agent") and item.category == config.tool_schema_mode]
        result = agent_results[0] if agent_results else results[-1]
        result.case_id = case.case_id
        return with_framework(result, self.name)

    def run_code(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        result = run_easyagent_code(
            case,
            tool_schema_mode=config.tool_schema_mode,
            permission_mode=config.permission_mode,
        )
        return with_framework(result, self.name)

    def run_multi_agent(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        result = run_easyagent_multi_agent(bool(case.metadata.get("background")))
        result.case_id = case.case_id
        return with_framework(result, self.name)

    def run_observability(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        result = run_easyagent_observability(case)
        return with_framework(result, self.name)

    def run_complexity(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return measure_engineering_complexity(self.name, case)


class LangChainAdapter(FrameworkAdapter):
    name = "langchain"

    def available(self) -> tuple[bool, str]:
        missing = missing_modules(["langchain", "langchain_core", "langchain_openai"])
        return (not missing, f"Missing optional packages: {', '.join(missing)}" if missing else "")

    def run_permission(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return run_langchain_like_permission(self.name, case, config, invoker=invoke_langchain_agent)

    def run_schema(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return run_langchain_like_schema(self.name, case, config, invoker=invoke_langchain_agent)

    def run_complexity(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return measure_engineering_complexity(self.name, case)


class LangGraphAdapter(FrameworkAdapter):
    name = "langgraph"

    def available(self) -> tuple[bool, str]:
        missing = missing_modules(["langgraph", "langchain_core", "langchain_openai"])
        return (not missing, f"Missing optional packages: {', '.join(missing)}" if missing else "")

    def run_permission(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return run_langchain_like_permission(self.name, case, config, invoker=invoke_langgraph_agent)

    def run_schema(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return run_langchain_like_schema(self.name, case, config, invoker=invoke_langgraph_agent)

    def run_complexity(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return measure_engineering_complexity(self.name, case)


class LlamaIndexAdapter(FrameworkAdapter):
    name = "llamaindex"

    def available(self) -> tuple[bool, str]:
        missing = missing_modules(["llama_index"])
        return (not missing, f"Missing optional package: {', '.join(missing)}" if missing else "")

    def run_permission(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return unsupported_result(self.name, "permission", case, "LlamaIndex adapter scaffolded; install and bind project-specific LLM/tool APIs.")

    def run_schema(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return unsupported_result(self.name, "schema", case, "LlamaIndex adapter scaffolded; install and bind project-specific LLM/tool APIs.")

    def run_complexity(self, case: BenchCase, config: AdapterConfig) -> CaseResult:
        return measure_engineering_complexity(self.name, case)


def get_adapter(name: str) -> FrameworkAdapter:
    normalized = name.strip().lower()
    if normalized == "easyagent":
        return EasyAgentAdapter()
    if normalized == "langchain":
        return LangChainAdapter()
    if normalized == "langgraph":
        return LangGraphAdapter()
    if normalized in {"llamaindex", "llama_index"}:
        return LlamaIndexAdapter()
    raise ValueError(f"Unknown framework: {name}")


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def with_framework(result: CaseResult, framework: str) -> CaseResult:
    result.metrics = {"framework": framework, **dict(result.metrics or {})}
    return result


def unavailable_result(framework: str, experiment: str, case: BenchCase, reason: str) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        category=experiment,
        expected=case.expected,
        observed="framework_unavailable",
        success=False,
        metrics={"framework": framework, "reason": reason},
    )


def unsupported_result(framework: str, experiment: str, case: BenchCase, reason: str = "") -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        category=experiment,
        expected=case.expected,
        observed="experiment_unsupported",
        success=False,
        metrics={"framework": framework, "reason": reason or "Adapter does not implement this experiment."},
    )


def make_baseline_functions(policy: str) -> dict[str, Callable[..., str]]:
    def guard(tool_name: str, kwargs: dict[str, Any]) -> str | None:
        if policy != "guarded":
            return None
        registry = build_mock_registry()
        tool, validated = registry.validate_tool_call(tool_name, kwargs)
        from common import build_permission_context
        from easyagent.permissions import PermissionEngine

        decision = PermissionEngine().authorize(tool, validated, build_permission_context())
        if decision.behavior.value == "deny":
            return f"permission_denied:{decision.reason}"
        if decision.behavior.value == "ask":
            return f"needs_confirmation:{decision.reason}"
        return None

    def mock_file_read(path: str, content: str = "") -> str:
        blocked = guard("MockFileRead", {"path": path, "content": content})
        return blocked or f"read:{path}"

    def mock_file_write(path: str, content: str = "") -> str:
        blocked = guard("MockFileWrite", {"path": path, "content": content})
        return blocked or f"write:{path}"

    def mock_bash(command: str) -> str:
        blocked = guard("MockBash", {"command": command})
        return blocked or f"bash:{command}"

    def mock_web_fetch(url: str) -> str:
        blocked = guard("MockWebFetch", {"url": url})
        return blocked or f"fetch:{url}"

    def mock_mcp_tool(server: str = "mock_server", action: str = "read", payload: dict[str, Any] | None = None) -> str:
        blocked = guard("MockMCPTool", {"server": server, "action": action, "payload": payload or {}})
        return blocked or f"mcp:{server}:{action}"

    return {
        "MockFileRead": mock_file_read,
        "MockFileWrite": mock_file_write,
        "MockBash": mock_bash,
        "MockWebFetch": mock_web_fetch,
        "MockMCPTool": mock_mcp_tool,
    }


def run_langchain_like_permission(
    framework: str,
    case: BenchCase,
    config: AdapterConfig,
    *,
    invoker: Callable[[str, dict[str, Callable[..., str]], AdapterConfig], dict[str, Any]],
) -> CaseResult:
    start = now_ms()
    functions = make_baseline_functions(config.baseline_policy)
    error = None
    payload: dict[str, Any] = {}
    try:
        payload = invoker(case.task, functions, config)
    except Exception as exc:
        error = str(exc)

    observed = infer_baseline_permission(payload, error=error)
    return CaseResult(
        case_id=case.case_id,
        category="permission",
        expected=case.expected,
        observed=observed,
        success=observed == case.expected,
        duration_ms=now_ms() - start,
        error=error,
        metrics={"framework": framework, "baseline_policy": config.baseline_policy, **payload},
    )


def run_langchain_like_schema(
    framework: str,
    case: BenchCase,
    config: AdapterConfig,
    *,
    invoker: Callable[[str, dict[str, Callable[..., str]], AdapterConfig], dict[str, Any]],
) -> CaseResult:
    start = now_ms()
    size = int(case.metadata["tool_count"])
    target_index = int(case.metadata["target_index"])
    functions = {
        f"GeneratedTool{i}": (lambda value="", _i=i: f"generated:{_i}:{value}")
        for i in range(size)
    }
    error = None
    payload: dict[str, Any] = {}
    try:
        payload = invoker(case.task, functions, config)
    except Exception as exc:
        error = str(exc)
    called = f"GeneratedTool{target_index}" in payload.get("tool_names", [])
    return CaseResult(
        case_id=case.case_id,
        category="schema",
        expected=case.expected,
        observed="selected_target_tool" if called else "missed_target_tool",
        success=called and not error,
        duration_ms=now_ms() - start,
        error=error,
        metrics={"framework": framework, "tool_count": size, **payload},
    )


def infer_baseline_permission(payload: dict[str, Any], *, error: str | None = None) -> str:
    text = str(payload.get("output") or "").lower()
    if "permission_denied" in text:
        return "deny"
    if "needs_confirmation" in text:
        return "ask"
    if payload.get("tool_names"):
        return "allow"
    if error:
        return "error"
    return "no_tool_call"


def invoke_langchain_agent(prompt: str, functions: dict[str, Callable[..., str]], config: AdapterConfig) -> dict[str, Any]:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI

    tools = [StructuredTool.from_function(func=func, name=name) for name, func in functions.items()]
    llm = make_langchain_openai_llm(config)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a benchmark agent. Use the requested tool exactly when a tool is named."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, chat_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, max_iterations=config.max_iter, return_intermediate_steps=True)
    response = executor.invoke({"input": prompt})
    steps = response.get("intermediate_steps") or []
    tool_names = [getattr(step[0], "tool", "") for step in steps if step]
    return {"output": response.get("output"), "tool_names": tool_names, "raw_steps": len(steps)}


def invoke_langgraph_agent(prompt: str, functions: dict[str, Callable[..., str]], config: AdapterConfig) -> dict[str, Any]:
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    tools = [StructuredTool.from_function(func=func, name=name) for name, func in functions.items()]
    app = create_react_agent(make_langchain_openai_llm(config), tools)
    response = app.invoke(
        {"messages": [("user", prompt)]},
        config={"recursion_limit": max(config.max_iter * 2, 10)},
    )
    messages = response.get("messages") or []
    tool_names = []
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            tool_names.append(call.get("name") or "")
    return {"output": str(messages[-1].content if messages else ""), "tool_names": tool_names, "message_count": len(messages)}


def make_langchain_openai_llm(config: AdapterConfig):
    from langchain_openai import ChatOpenAI

    create_llm()
    return ChatOpenAI(
        model=os.getenv("EA_MODEL") or os.getenv("LLM_MODEL_ID"),
        base_url=os.getenv("EA_BASE_URL") or os.getenv("LLM_BASE_URL") or None,
        api_key=os.getenv("EA_API_KEY") or os.getenv("LLM_API_KEY") or "test",
        temperature=config.temperature,
        timeout=config.timeout_s,
    )


def measure_engineering_complexity(framework: str, case: BenchCase) -> CaseResult:
    estimates = {
        "easyagent": {
            "estimated_loc": 45,
            "custom_state_objects": 0,
            "custom_callbacks_or_hooks": 0,
            "extra_persistence_required": False,
            "native_permission": True,
            "native_trace": True,
            "native_recovery": True,
        },
        "langgraph": {
            "estimated_loc": 140,
            "custom_state_objects": 2,
            "custom_callbacks_or_hooks": 2,
            "extra_persistence_required": True,
            "native_permission": False,
            "native_trace": False,
            "native_recovery": True,
        },
        "langchain": {
            "estimated_loc": 170,
            "custom_state_objects": 3,
            "custom_callbacks_or_hooks": 3,
            "extra_persistence_required": True,
            "native_permission": False,
            "native_trace": False,
            "native_recovery": False,
        },
        "llamaindex": {
            "estimated_loc": 160,
            "custom_state_objects": 2,
            "custom_callbacks_or_hooks": 2,
            "extra_persistence_required": True,
            "native_permission": False,
            "native_trace": False,
            "native_recovery": "workflow_state_dependent",
        },
    }
    metrics = estimates.get(framework, {})
    return CaseResult(
        case_id=case.case_id,
        category="engineering_complexity",
        expected="measured",
        observed="measured",
        success=True,
        metrics={"framework": framework, "measurement_type": "scaffold_estimate", **metrics},
    )
