"""History message assembler interfaces and default implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from agent import BasicAgent
from core.providers import create_codec
from core.request_input import ReplayRequestInput

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
    def build_start_messages(self, agent:BasicAgent, query: str)->Any:
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

    def build_start_messages(self, agent:BasicAgent, query: str) -> ReplayRequestInput:
        system_prompt = agent.get_enhanced_prompt()
        self._ensure_replay_history(agent)
        provider_name = getattr(agent.llm, "provider_name", None)
        if agent.context_manager is not None:
            request_input = agent.context_manager.build_request_input(
                query=query,
                replay_history=agent.replay_history,
                provider_name=provider_name,
                system_prompt=system_prompt,
                include_query=True,
                tools=agent.tool_registry.get_openai_tools() if agent.tool_registry is not None else None,
                reasoning=agent.reasoning,
            )
            return request_input

        request_input = ReplayRequestInput(
            provider_name=provider_name,
            replay_history=list(agent.replay_history),
            system_prompt=system_prompt,
        )
        request_input.extend_replay(agent.llm.query_to_replay(query))
        return request_input
