"""Filtering and cleaning policies for training-data export."""

from __future__ import annotations

from abc import ABC

from observability import AgentInvoke


class TrainingDataFilter(ABC):
    """Select and optionally sanitize one finalized AgentInvoke."""

    def accept(self, invoke: AgentInvoke) -> bool:
        return True

    def transform(self, invoke: AgentInvoke) -> AgentInvoke:
        return invoke.model_copy(deep=True)

    def apply(self, invoke: AgentInvoke) -> AgentInvoke | None:
        candidate = invoke.model_copy(deep=True)
        if not self.accept(candidate):
            return None
        return self.transform(candidate)


class SuccessfulAgentInvokeFilter(TrainingDataFilter):
    """Default policy: export only completed, successful Agent invocations."""

    def accept(self, invoke: AgentInvoke) -> bool:
        return (
            invoke.stats.success
            and invoke.stats.status == "success"
            and invoke.stats.ended_at is not None
        )


__all__ = ["SuccessfulAgentInvokeFilter", "TrainingDataFilter"]
