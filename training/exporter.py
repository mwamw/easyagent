"""Training dataset exporter backed exclusively by observability records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from observability import (
    AgentInvoke,
    BaseObservabilityManager,
    BaseObservabilityStore,
)

from .filters import SuccessfulAgentInvokeFilter, TrainingDataFilter
from .models import TrainingDataFormat, TrainingExportReport


class TrainingExporter:
    """Build Step SFT, Trace SFT, and Agentic Rollout data from one store."""

    def __init__(self, source: BaseObservabilityManager | BaseObservabilityStore):
        if not isinstance(source, (BaseObservabilityManager, BaseObservabilityStore)):
            raise TypeError(
                "source must extend BaseObservabilityManager or BaseObservabilityStore"
            )
        self.source = source

    @classmethod
    def from_agent(cls, agent: Any) -> "TrainingExporter":
        observability = getattr(agent, "observability", None)
        if observability is None:
            raise RuntimeError(
                "Agent 未启用可观测模块，请先调用 agent.with_observability()。"
            )
        return cls(observability)

    def _filtered_records(
        self,
        data_filter: TrainingDataFilter | None,
    ) -> tuple[list[AgentInvoke], list[AgentInvoke]]:
        policy = data_filter or SuccessfulAgentInvokeFilter()
        source = self.source.list()
        accepted: list[AgentInvoke] = []
        for invoke in source:
            candidate = policy.apply(invoke)
            if candidate is not None:
                accepted.append(candidate)
        return source, accepted

    def build(
        self,
        data_format: TrainingDataFormat | str,
        *,
        data_filter: TrainingDataFilter | None = None,
    ) -> list[dict[str, Any]]:
        resolved = TrainingDataFormat(data_format)
        source, accepted = self._filtered_records(data_filter)
        if resolved is TrainingDataFormat.STEP_SFT:
            return self._build_step_sft(accepted)
        if resolved is TrainingDataFormat.TRACE_SFT:
            return self._build_trace_sft(accepted)
        return self._build_agentic_rollouts(accepted)

    def export(
        self,
        output_dir: str | Path,
        *,
        formats: Iterable[TrainingDataFormat | str] | None = None,
        data_filter: TrainingDataFilter | None = None,
    ) -> TrainingExportReport:
        selected_formats = formats if formats is not None else (
            TrainingDataFormat.STEP_SFT,
            TrainingDataFormat.TRACE_SFT,
            TrainingDataFormat.AGENTIC_ROLLOUT,
        )
        resolved_formats = [
            TrainingDataFormat(item)
            for item in selected_formats
        ]
        source, accepted = self._filtered_records(data_filter)
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        report = TrainingExportReport(
            source_records=len(source),
            accepted_records=len(accepted),
            rejected_records=len(source) - len(accepted),
        )
        for data_format in dict.fromkeys(resolved_formats):
            if data_format is TrainingDataFormat.STEP_SFT:
                records = self._build_step_sft(accepted)
            elif data_format is TrainingDataFormat.TRACE_SFT:
                records = self._build_trace_sft(accepted)
            else:
                records = self._build_agentic_rollouts(accepted)
            path = directory / f"{data_format.value}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.files[data_format.value] = str(path)
            report.counts[data_format.value] = len(records)
        return report

    @staticmethod
    def _build_step_sft(invokes: list[AgentInvoke]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for agent_invoke in invokes:
            for llm_invoke in agent_invoke.llm_invokes:
                if not llm_invoke.stats.success:
                    continue
                payload = llm_invoke.to_dict()
                input_messages = payload["input"]
                output_messages = payload["output"]
                records.append(
                    {
                        "schema_version": "easyagent.training.v1",
                        "format": TrainingDataFormat.STEP_SFT.value,
                        "source_agent_invoke_id": agent_invoke.invoke_id,
                        "source_llm_invoke_id": llm_invoke.invoke_id,
                        "input": {
                            "messages": input_messages,
                            "tools": payload["tools"],
                            "options": payload["options"],
                        },
                        "output": output_messages,
                        "messages": input_messages + output_messages,
                        "stats": payload["stats"],
                        "metadata": payload["metadata"],
                    }
                )
        return records

    @staticmethod
    def _build_trace_sft(invokes: list[AgentInvoke]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for invoke in invokes:
            payload = invoke.to_dict()
            records.append({
                "schema_version": "easyagent.training.v1",
                "format": TrainingDataFormat.TRACE_SFT.value,
                "source_agent_invoke_id": invoke.invoke_id,
                "query": invoke.query,
                "trace": payload["trace"],
                "output": payload["output"],
                "stats": payload["stats"],
                "metadata": payload["metadata"],
            })
        return records

    @staticmethod
    def _build_agentic_rollouts(
        accepted: list[AgentInvoke],
    ) -> list[dict[str, Any]]:
        by_id = {invoke.invoke_id: invoke for invoke in accepted}
        children: dict[str, list[AgentInvoke]] = {}
        for invoke in accepted:
            if invoke.parent_invoke_id:
                children.setdefault(invoke.parent_invoke_id, []).append(invoke)

        roots = [
            invoke
            for invoke in accepted
            if not invoke.parent_invoke_id or invoke.parent_invoke_id not in by_id
        ]

        def collect(root: AgentInvoke) -> list[AgentInvoke]:
            ordered: list[AgentInvoke] = []
            pending = [root]
            seen: set[str] = set()
            while pending:
                current = pending.pop(0)
                if current.invoke_id in seen:
                    continue
                seen.add(current.invoke_id)
                ordered.append(current)
                pending.extend(
                    sorted(
                        children.get(current.invoke_id, []),
                        key=lambda item: item.stats.started_at,
                    )
                )
            return ordered

        rollouts: list[dict[str, Any]] = []
        for root in roots:
            trajectory = collect(root)
            rollouts.append(
                {
                    "schema_version": "easyagent.training.v1",
                    "format": TrainingDataFormat.AGENTIC_ROLLOUT.value,
                    "root_agent_invoke_id": root.invoke_id,
                    "agent_invokes": [invoke.to_dict() for invoke in trajectory],
                    "stats": {
                        "agent_invokes": len(trajectory),
                        "successful_agent_invokes": sum(
                            1 for invoke in trajectory if invoke.stats.success
                        ),
                        "llm_invokes": sum(
                            len(invoke.llm_invokes) for invoke in trajectory
                        ),
                        "tool_calls": sum(
                            invoke.stats.tool_calls for invoke in trajectory
                        ),
                        "total_tokens": sum(
                            invoke.stats.total_tokens for invoke in trajectory
                        ),
                    },
                    "metadata": {
                        "agent_invoke_ids": [
                            invoke.invoke_id for invoke in trajectory
                        ]
                    },
                }
            )
        return rollouts


__all__ = ["TrainingExporter"]
