"""Context contract with framework-owned canonical conversation history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, TYPE_CHECKING

from core.cache_policy import PromptCachePolicy
from core.history import CanonicalMessage, coerce_canonical_message
from core.request_input import ReplayRequestInput
from runtime.metamessage.models import MetaMessageInjection
from prompt.system_prompt import PromptBlock
from .history import ConversationHistory
from .models import (
    ContextCompactionResult,
    ContextUsage,
    ContextUsageReport,
    ModelRequestOptions,
    ModelUsage,
    ProviderReplayHistory,
    ToolDefinitions,
)

if TYPE_CHECKING:
    from core.llm import EasyLLM
    from runtime.metamessage.manager import MetaMessageManager
    from prompt.base import PromptBuildContext


class BaseContextManager(ABC):
    """Context Runtime contract coordinated explicitly by an Agent Executor.

    The framework supplies canonical history behavior. Implementations define
    input assembly, request sizing, compaction policy and input validation.
    """

    def __init__(self, history: ConversationHistory | None = None):
        self._history = history or ConversationHistory()
        self._llm: EasyLLM | None = None
        self._metamessages: MetaMessageManager | None = None
        self._owner = None

    def bind(
        self,
        llm: EasyLLM,
        *,
        metamessages: MetaMessageManager | None = None,
        owner: object | None = None,
    ) -> "BaseContextManager":
        if owner is not None:
            if self._owner is not None and self._owner is not owner:
                raise RuntimeError("A stateful Context cannot be shared between Agents")
            self._owner = owner
        self._llm = llm
        self._metamessages = metamessages
        return self

    def _require_llm(self) -> EasyLLM:
        if self._llm is None:
            raise RuntimeError("Context must be bound to an LLM")
        return self._llm

    @property
    def history(self) -> ConversationHistory:
        return self._history

    @abstractmethod
    def build_input(
        self,
        prompt_blocks: Sequence[PromptBlock],
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        query: str,
        cache_policy: PromptCachePolicy | None,
        replay_history: ProviderReplayHistory | None = None,
    ) -> ReplayRequestInput:
        """Assemble the complete provider-neutral model request."""
        raise NotImplementedError

    async def abuild_input(
        self,
        prompt_blocks: Sequence[PromptBlock],
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        query: str,
        cache_policy: PromptCachePolicy | None,
        replay_history: ProviderReplayHistory | None = None,
    ) -> ReplayRequestInput:
        import asyncio
        return await asyncio.to_thread(
            self.build_input, prompt_blocks, tools=tools, options=options,
            query=query, cache_policy=cache_policy,
            replay_history=replay_history,
        )

    @abstractmethod
    def calculate_usage(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
    ) -> ContextUsage:
        """Calculate the complete request size and available context budget."""
        raise NotImplementedError

    async def acalculate_usage(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
    ) -> ContextUsage:
        import asyncio
        return await asyncio.to_thread(
            self.calculate_usage, request_input, tools=tools, options=options,
        )

    @abstractmethod
    def should_compact(self, usage: ContextUsage) -> bool:
        """Decide whether the measured request requires compaction."""
        raise NotImplementedError

    @abstractmethod
    def compact_context(
        self,
        request_input: ReplayRequestInput,
        usage: ContextUsage,
        *,
        canonical_history: list[CanonicalMessage],
        replay_history: ProviderReplayHistory,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        max_tokens: int | None = None,
    ) -> ContextCompactionResult:
        """Produce a candidate compacted history without committing it."""
        raise NotImplementedError

    async def acompact_context(
        self,
        request_input: ReplayRequestInput,
        usage: ContextUsage,
        *,
        canonical_history: list[CanonicalMessage],
        replay_history: ProviderReplayHistory,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        max_tokens: int | None = None,
    ) -> ContextCompactionResult:
        import asyncio
        return await asyncio.to_thread(
            self.compact_context, request_input, usage,
            canonical_history=canonical_history,
            replay_history=replay_history, tools=tools, options=options,
            max_tokens=max_tokens,
        )

    @abstractmethod
    def validate_input(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        history_epoch: int | None = None,
    ) -> None:
        """Reject an input that still violates this Context implementation's rules."""
        raise NotImplementedError

    async def avalidate_input(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        history_epoch: int | None = None,
    ) -> None:
        import asyncio
        await asyncio.to_thread(
            self.validate_input, request_input, tools=tools, options=options,
            history_epoch=history_epoch,
        )

    def commit_compaction(self, result: ContextCompactionResult) -> None:
        """Commit a successful compaction after all request hooks have passed."""
        if result.was_compacted:
            self.replace(result.canonical_history)

    def observe_prepared_input(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        query: str,
    ) -> None:
        """Observe the accepted request after any persistent compaction commits."""

    async def aobserve_prepared_input(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: ToolDefinitions,
        options: ModelRequestOptions,
        query: str,
    ) -> None:
        import asyncio
        await asyncio.to_thread(
            self.observe_prepared_input, request_input, tools=tools,
            options=options, query=query,
        )

    def process_meta_messages(self) -> list[MetaMessageInjection]:
        """Apply pending MetaMessages before collecting the next request."""
        if self._metamessages is not None:
            return self._metamessages.flush()
        return []

    async def aprocess_meta_messages(self) -> list[MetaMessageInjection]:
        import asyncio
        return await asyncio.to_thread(self.process_meta_messages)

    def observe_model_result(
        self,
        usage: ModelUsage,
        response_replay: ProviderReplayHistory,
        *,
        use_output: bool,
    ) -> None:
        """Optional provider-usage feedback; custom contexts need no token counter."""

    @property
    def history_epoch(self) -> int:
        """Generation changed by destructive history replacement, not appends."""
        return 0

    def validation_epoch(self, compaction: ContextCompactionResult | None) -> int:
        """Return this implementation's generation for validating a candidate input."""
        return self.history_epoch

    @property
    def canonical(self) -> list[CanonicalMessage]:
        return self._history.messages

    @property
    def replay(self) -> ProviderReplayHistory:
        return self._require_llm().canonical_to_replay_history(self.canonical)

    def append_query(self, query: str) -> None:
        self._history.extend(self._require_llm().query_to_canonical(str(query)))

    def append_response(self, response: Any, *, include_reasoning: bool = False) -> None:
        self._history.extend(
            self._require_llm().response_to_canonical(
                response, include_reasoning=include_reasoning
            )
        )

    def append_assistant(self, *, content: str | None = None,
                         tool_calls: list[dict[str, Any]] | None = None,
                         thinking: str | None = None) -> None:
        self._history.extend(
            self._require_llm().assistant_message_to_canonical(
                content=content, tool_calls=tool_calls, thinking=thinking,
            )
        )

    def append_tool_result(self, content: str, tool_id: str, tool_name: str) -> None:
        self._history.extend(
            self._require_llm().tool_result_to_canonical(
                content, tool_id, tool_name
            )
        )

    @abstractmethod
    def append_tool_context(self, tool_name: str, value: object) -> None:
        """Expose ephemeral tool-produced context to a future model request."""
        raise NotImplementedError

    def add(self, message: CanonicalMessage | dict[str, Any] | object) -> None:
        llm = self._require_llm()
        canonical = llm.history_entry_to_canonical(message)
        if not canonical:
            value = coerce_canonical_message(message)
            canonical = [value] if value is not None else []
        if not canonical:
            raise TypeError(f"Unsupported history message: {type(message).__name__}")
        self._history.extend(canonical)

    def add_many(self, messages: Sequence[CanonicalMessage | dict[str, Any] | object]) -> None:
        for message in messages:
            self.add(message)

    def replace(self, messages: Sequence[CanonicalMessage | dict[str, Any] | object]) -> None:
        llm = self._require_llm()
        normalized = []
        for message in messages:
            value = coerce_canonical_message(message)
            if value is not None:
                normalized.append(value)
            else:
                normalized.extend(llm.history_entry_to_canonical(message))
        self._history.replace(normalized)

    def clear(self) -> None:
        self._history.clear()

    def rebuild_replay(self) -> ProviderReplayHistory:
        """Contexts without a replay cache may simply return the current projection."""
        return self.replay

    def export_state(self) -> dict[str, Any]:
        return {
            "providerName": getattr(self._llm, "provider_name", None),
            **self._history.export_state(),
        }

    def validate_state(self, state: dict[str, Any] | None) -> None:
        self._history.validate_state(state)

    def restore_state(self, state: dict[str, Any] | None) -> None:
        self.validate_state(state)
        self._history.restore_state(state)

    def get_prompt_blocks(self, context: "PromptBuildContext") -> Sequence[PromptBlock]:
        return []

    def close(self) -> None:
        """Release Context-owned resources, never the borrowed LLM."""

    def get_usage(self, prompt_blocks: Sequence[PromptBlock], *, tools: ToolDefinitions = None,
                  reasoning: dict[str, Any] | None = None,
                  cache_policy: PromptCachePolicy | None = None) -> ContextUsageReport:
        options = {"reasoning": reasoning}
        request = self.build_input(prompt_blocks, tools=tools, options=options,
                                   query=self.last_query, cache_policy=cache_policy)
        usage = self.calculate_usage(request, tools=tools, options=options)
        return {
            "estimatedRequestTokens": usage.estimated_tokens,
            "localEstimatedRequestTokens": usage.local_estimated_tokens,
            "request_estimate_source": usage.source,
            "requestEstimate": {"estimatedRequestTokens": usage.estimated_tokens,
                                "source": usage.source,
                                "localEstimatedRequestTokens": usage.local_estimated_tokens,
                                "metadata": usage.metadata},
            "maxTokens": usage.max_tokens,
            "budget": {"maxTokens": usage.max_tokens, "remainingTokens": usage.remaining_tokens},
            "canonicalMessages": len(self.canonical), "replayMessages": len(self.replay),
            "provider": self._require_llm().provider_name,
        }

    @property
    def revision(self) -> int:
        return self._history.revision

    @property
    def last_query(self) -> str:
        return ""

    @property
    def usage_revision(self) -> int:
        return 0

    def remove_by_metadata(self, key: str, value: object) -> bool:
        return self._history.remove_by_metadata(key, value)

    def contains_metadata(self, key: str, value: object) -> bool:
        return self._history.contains_metadata(key, value)

    def __len__(self) -> int:
        return len(self._history)
