"""History message assembler interfaces and default implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from core.Message import SystemMessage, UserMessage

logger = logging.getLogger(__name__)


class BaseHistoryMessageAssembler(ABC):
    """Assembles start messages and history injection strategy for BasicAgent."""

    @abstractmethod
    def use_context_history(self, agent: Any) -> bool:
        pass

    @abstractmethod
    def context_include_history(self, agent: Any) -> bool:
        pass

    @abstractmethod
    def append_runtime_history(self, agent: Any, messages: list[Any]) -> None:
        pass

    @abstractmethod
    def build_start_messages(self, agent: Any, query: str) -> list[Any]:
        pass


class DefaultHistoryMessageAssembler(BaseHistoryMessageAssembler):
    """Default history assembler that preserves current BasicAgent behavior."""

    def use_context_history(self, agent: Any) -> bool:
        return bool(agent.history_via_context_manager and agent.context_manager)

    def context_include_history(self, agent: Any) -> bool:
        return self.use_context_history(agent)

    def append_runtime_history(self, agent: Any, messages: list[Any]) -> None:
        if self.use_context_history(agent):
            return
        for message in agent.history:
            messages.append(message)

    def build_start_messages(self, agent: Any, query: str) -> list[Any]:
        system_prompt = agent.get_enhanced_prompt()

        if agent.context_manager is not None:
            try:
                messages = agent.context_manager.build_messages(
                    query=query,
                    history=agent.history,
                    system_prompt=system_prompt,
                    include_history=True,
                    include_query=True,
                )
                if agent.context_manager.last_history_was_compacted:
                    agent.history = list(agent.context_manager.last_compacted_history)
                return messages
            except Exception as exc:
                logger.warning(f"ContextManager 构建 messages 失败，回退默认拼接: {exc}")

        messages: list[Any] = [SystemMessage(system_prompt)]
        self.append_runtime_history(agent, messages)
        messages.append(UserMessage(query))
        return messages
