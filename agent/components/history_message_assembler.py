"""History message assembler interfaces and default implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from agent import BasicAgent
from core.Message import SystemMessage, UserMessage
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
    def build_start_messages(self, agent:BasicAgent, query: str) -> list[Any]:
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

    def build_start_messages(self, agent:BasicAgent, query: str) ->list[Any]:
        system_prompt = agent.get_enhanced_prompt()
        self._ensure_replay_history(agent)
        request_ready_checker = agent.llm._get_codec().is_request_ready_message

        if agent.context_manager is not None:
            try:
                request_input = agent.context_manager.build_request_input(
                    query=query,
                    history=agent.history,
                    replay_history=agent.replay_history,
                    history_converter=agent.prepare_replay_history,
                    message_converter=agent.prepare_replay_history,
                    request_ready_checker=request_ready_checker,
                    provider_name=getattr(agent.llm, "provider_name", None),
                    system_prompt=system_prompt,
                    include_history=True,
                    include_query=True,
                )
                if agent.context_manager.last_history_was_compacted:
                    agent.history = list(agent.context_manager.last_compacted_history)
                return request_input
            except Exception as exc:
                logger.warning(f"ContextManager 构建 messages 失败，回退默认拼接: {exc}")

        request_input = ReplayRequestInput(
            provider_name=getattr(agent.llm, "provider_name", None),
            replay_history=list(agent.replay_history),
            system_prompt=system_prompt,
            message_converter=agent.prepare_replay_history,
            request_ready_checker=request_ready_checker,
        )
        request_input.append(UserMessage(query))
        return request_input
