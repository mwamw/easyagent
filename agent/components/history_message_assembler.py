"""History message assembler interfaces and default implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from agent import BasicAgent
from core.request_input import ReplayRequestInput
from core.cache_policy import build_cache_signature
from core.request_compiler import compile_prompt_blocks

logger = logging.getLogger(__name__)


class BaseHistoryMessageAssembler(ABC):
    """Assembles start messages and history injection strategy for BasicAgent."""

    @abstractmethod
    def use_context_history(self, agent:BasicAgent) -> bool:
        pass

    @abstractmethod
    def context_include_history(self, agent:BasicAgent) -> bool:
        pass

    @abstractmethod
    def append_runtime_history(self, agent:BasicAgent, messages: list[Any]) -> None:
        pass

    @abstractmethod
    def build_start_messages(
        self,
        agent:BasicAgent,
        query: str,
        *,
        include_query: bool = True,
        extra_replay_entries: list[Any] | None = None,
    )->Any:
        pass


class DefaultHistoryMessageAssembler(BaseHistoryMessageAssembler):
    """Default history assembler that preserves current BasicAgent behavior."""

    @staticmethod
    def _ensure_replay_history(agent:BasicAgent) -> None:
        current_provider = getattr(getattr(agent, "llm", None), "provider_name", None)
        if not agent.history:
            return
        if getattr(agent, "replay_history_provider_name", None) != current_provider:
            raise RuntimeError(
                "检测到 provider 已切换但 replay_history 尚未同步。请通过 agent.change_model(...) 切换模型。"
            )

    def use_context_history(self, agent:BasicAgent) -> bool:
        return bool(agent.history_via_context_manager and agent.context_manager)

    def context_include_history(self, agent:BasicAgent) -> bool:
        return self.use_context_history(agent)

    def append_runtime_history(self, agent:BasicAgent, messages: list[Any]) -> None:
        if self.use_context_history(agent):
            return
        self._ensure_replay_history(agent)
        for message in agent.replay_history:
            messages.append(message)

    def build_start_messages(
        self,
        agent:BasicAgent,
        query: str,
        *,
        include_query: bool = True,
        extra_replay_entries: list[Any] | None = None,
    ) -> ReplayRequestInput:
        tools_payload = None
        if agent.tool_registry is not None:
            try:
                tools_payload = agent.get_provider_tools()
            except Exception:
                tools_payload = None
        compiled_prompt = compile_prompt_blocks(
            agent.get_system_prompt_blocks(),
            cache_policy=getattr(agent.config, "cache_policy", None),
            cache_dynamic_memory=bool(getattr(agent.config, "cache_dynamic_memory", False)),
            cache_dynamic_mailbox=bool(getattr(agent.config, "cache_dynamic_mailbox", False)),
            cache_turn_skills=bool(getattr(agent.config, "cache_turn_skills", False)),
        )
        self._ensure_replay_history(agent)
        provider_name = getattr(agent.llm, "provider_name", None)
        cache_metadata = dict(compiled_prompt.metadata or {})
        google_cached_content_name = getattr(getattr(agent, "config", None), "google_cached_content_name", None)
        if provider_name in {"google_native", "gemini_native"} and google_cached_content_name:
            cache_metadata["googleCachedContent"] = str(google_cached_content_name)
        if agent.context_manager is not None:
            request_input = agent.context_manager.build_request_input(
                query=query,
                replay_history=agent.replay_history,
                provider_name=provider_name,
                system_prompt=compiled_prompt.system_prompt,
                system_prompt_blocks=compiled_prompt.system_prompt_blocks,
                runtime_reminder_blocks=compiled_prompt.runtime_reminder_blocks,
                dynamic_tail_blocks=compiled_prompt.dynamic_tail_blocks,
                on_demand_expansion_blocks=compiled_prompt.on_demand_expansion_blocks,
                cache_policy=compiled_prompt.cache_policy,
                cache_metadata=cache_metadata,
                include_query=include_query,
                extra_replay_entries=extra_replay_entries,
                tools=tools_payload,
                reasoning=agent.reasoning,
            )
            request_input.cache_signature = build_cache_signature(
                provider=provider_name,
                model=getattr(agent.llm, "model", None),
                system_blocks=[
                    *request_input.system_prompt_blocks,
                    *request_input.runtime_reminder_blocks,
                ],
                tools=tools_payload,
                reasoning=agent.reasoning,
                extra={
                    "runtime_reminders": [
                        block.to_dict() for block in request_input.runtime_reminder_blocks
                        if block.cacheable
                    ],
                },
                cache_policy=request_input.cache_policy,
            ).to_dict()
            return request_input

        request_input = ReplayRequestInput(
            provider_name=provider_name,
            replay_history=list(agent.replay_history),
            system_prompt=compiled_prompt.system_prompt,
            system_prompt_blocks=compiled_prompt.system_prompt_blocks,
            runtime_reminder_blocks=compiled_prompt.runtime_reminder_blocks,
            dynamic_tail_blocks=compiled_prompt.dynamic_tail_blocks,
            on_demand_expansion_blocks=compiled_prompt.on_demand_expansion_blocks,
            cache_policy=compiled_prompt.cache_policy,
            cache_metadata=cache_metadata,
        )
        request_input.apply_runtime_layers()
        if extra_replay_entries:
            request_input.extend_replay(extra_replay_entries)
        if include_query and query:
            request_input.extend_replay(agent.llm.query_to_replay(query))
        request_input.cache_signature = build_cache_signature(
            provider=provider_name,
            model=getattr(agent.llm, "model", None),
            system_blocks=[
                *request_input.system_prompt_blocks,
                *request_input.runtime_reminder_blocks,
            ],
            tools=tools_payload,
            reasoning=agent.reasoning,
            extra={
                "runtime_reminders": [
                    block.to_dict() for block in request_input.runtime_reminder_blocks
                    if block.cacheable
                ],
            },
            cache_policy=request_input.cache_policy,
        ).to_dict()
        return request_input
