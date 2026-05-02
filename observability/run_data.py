"""Structured run/eval/training data models for EasyAgent observability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


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
class RunOutcomeReport:
    run_id: str
    status: str
    success: bool
    failure_stage: Optional[str] = None
    root_cause_tags: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tests_attempted: list[str] = field(default_factory=list)
    tests_passed: list[str] = field(default_factory=list)
    tests_failed: list[str] = field(default_factory=list)
    user_approval_count: int = 0
    user_deny_count: int = 0
    notes: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class AgentRunRecord:
    run_id: str
    session_id: Optional[str]
    agent_name: Optional[str]
    turn_id: Optional[str]
    query: str
    mode: str
    stream: bool
    success: bool
    status: str
    started_at: Optional[str]
    ended_at: Optional[str]
    duration_ms: Optional[float]
    provider_name: Optional[str] = None
    model: Optional[str] = None
    output_preview: str = ""
    final_output: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    llm_requests: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    cache_breaks: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    outcome: Optional[RunOutcomeReport] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.outcome is not None:
            payload["outcome"] = self.outcome.to_dict()
        return _json_safe(payload)


@dataclass
class EvalTrace:
    run_id: str
    query: str
    success: bool
    status: str
    final_output: str
    tool_calls: int
    llm_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_ms: Optional[float]
    prompt_tokens_cached: int = 0
    prompt_tokens_uncached: int = 0
    cache_hit_ratio: Optional[float] = None
    tools_used: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    cache_break_reasons: list[str] = field(default_factory=list)
    failure_stage: Optional[str] = None
    root_cause_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class TrainingExample:
    example_id: str
    run_id: str
    example_type: str
    input: dict[str, Any]
    target: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class SFTDatasetRecord:
    record_id: str
    run_id: str
    source_example_id: str
    example_type: str
    prompt: str
    completion: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    input: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class PreferencePairRecord:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_run_id: str
    rejected_run_id: str
    chosen_score: float
    rejected_score: float
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


__all__ = [
    "AgentRunRecord",
    "EvalTrace",
    "PreferencePairRecord",
    "RunOutcomeReport",
    "SFTDatasetRecord",
    "TrainingExample",
]
