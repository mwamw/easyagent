"""Framework-level values exchanged by Context preparation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, TypedDict
from core.history import CanonicalMessage


ToolDefinitions: TypeAlias = list[dict[str, object]] | None
ModelRequestOptions: TypeAlias = dict[str, object]
ProviderReplayHistory: TypeAlias = list[object]


class ModelUsage(TypedDict, total=False):
    """Provider-neutral usage fields observed by Context after a model call."""

    inputTokens: int | None
    outputTokens: int | None
    totalTokens: int | None
    contextInputTokens: int | None
    contextTotalTokens: int | None
    reasoningTokens: int | None
    cachedInputTokens: int | None
    cacheReadTokens: int | None
    cacheCreationTokens: int | None
    costUsd: float | None
    usageSource: Literal["provider", "estimated"]
    cacheUsageSemantics: str
    supportsCacheUsageFields: bool


@dataclass(frozen=True)
class ContextUsage:
    """Measured or estimated size of one fully assembled model request."""

    estimated_tokens: int
    max_tokens: int | None
    local_estimated_tokens: int | None = None
    source: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.estimated_tokens)

    @property
    def over_budget(self) -> bool:
        return self.max_tokens is not None and self.estimated_tokens > self.max_tokens


@dataclass
class ContextCompactionResult:
    """Framework result required from a Context compaction implementation."""

    canonical_history: list[CanonicalMessage]
    replay_history: ProviderReplayHistory
    was_compacted: bool
    compaction_possible: bool
    tokens_before: int
    tokens_after: int
    budget: int
    metadata: dict[str, object] = field(default_factory=dict)
