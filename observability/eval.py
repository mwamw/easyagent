"""Minimal offline eval harness for EasyAgent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    return str(value)


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_output_contains: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    max_tool_calls: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class EvalResult:
    case_id: str
    query: str
    success: bool
    status: str
    score: float
    run_id: Optional[str] = None
    final_output: str = ""
    tools_used: list[str] = field(default_factory=list)
    tool_calls: int = 0
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_hit_ratio: Optional[float] = None
    failure_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class OfflineEvalHarness:
    """Runs replayable eval cases against fresh agent instances."""

    def run_cases(
        self,
        agent_factory: Callable[[], Any],
        cases: list[EvalCase],
        *,
        redact: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for case in cases:
            agent = agent_factory()
            try:
                agent.invoke(case.query)
                trace = agent.export_eval_trace(redact=redact) or {}
                result = self._score_case(case, trace)
                results.append(result.to_dict())
            finally:
                close = getattr(agent, "close", None)
                if callable(close):
                    try:
                        close(close_worktree=False)
                    except TypeError:
                        close()
        return results

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {
                "cases": 0,
                "successes": 0,
                "successRate": None,
                "avgScore": None,
                "avgToolCalls": None,
                "avgLlmRequests": None,
                "avgCacheHitRatio": None,
            }
        scores = [float(item.get("score") or 0.0) for item in results]
        tool_calls = [int(item.get("tool_calls") or 0) for item in results]
        llm_requests = [int(item.get("llm_requests") or 0) for item in results]
        cache_ratios = [
            float(item["cache_hit_ratio"])
            for item in results
            if item.get("cache_hit_ratio") is not None
        ]
        success_count = sum(1 for item in results if bool(item.get("success")))
        count = len(results)
        return {
            "cases": count,
            "successes": success_count,
            "successRate": float(success_count) / float(count),
            "avgScore": sum(scores) / float(count),
            "avgToolCalls": sum(tool_calls) / float(count),
            "avgLlmRequests": sum(llm_requests) / float(count),
            "avgCacheHitRatio": (
                sum(cache_ratios) / float(len(cache_ratios))
                if cache_ratios
                else None
            ),
        }

    def _score_case(self, case: EvalCase, trace: dict[str, Any]) -> EvalResult:
        output = str(trace.get("final_output") or "")
        tools_used = [str(item) for item in list(trace.get("tools_used") or []) if str(item).strip()]
        tool_calls = int(trace.get("tool_calls") or 0)
        failures: list[str] = []
        matched_checks = 0
        total_checks = 0

        for snippet in case.expected_output_contains:
            total_checks += 1
            if snippet and snippet in output:
                matched_checks += 1
            else:
                failures.append(f"missing_output:{snippet}")

        for tool_name in case.expected_tools:
            total_checks += 1
            if tool_name in tools_used:
                matched_checks += 1
            else:
                failures.append(f"missing_tool:{tool_name}")

        if case.max_tool_calls is not None:
            total_checks += 1
            if tool_calls <= int(case.max_tool_calls):
                matched_checks += 1
            else:
                failures.append(f"tool_calls_exceeded:{tool_calls}")

        if total_checks == 0:
            success = bool(trace.get("success"))
            score = 1.0 if success else 0.0
        else:
            success = matched_checks == total_checks
            score = float(matched_checks) / float(total_checks)

        return EvalResult(
            case_id=case.case_id,
            query=case.query,
            success=success,
            status="success" if success else "failed",
            score=score,
            run_id=trace.get("run_id"),
            final_output=output,
            tools_used=tools_used,
            tool_calls=tool_calls,
            llm_requests=int(trace.get("llm_requests") or 0),
            input_tokens=int(trace.get("input_tokens") or 0),
            output_tokens=int(trace.get("output_tokens") or 0),
            total_tokens=int(trace.get("total_tokens") or 0),
            cache_hit_ratio=trace.get("cache_hit_ratio"),
            failure_reasons=failures,
            metadata={
                "caseMetadata": _json_safe(case.metadata),
                "traceMetadata": _json_safe(trace.get("metadata") or {}),
            },
        )


__all__ = ["EvalCase", "EvalResult", "OfflineEvalHarness"]
