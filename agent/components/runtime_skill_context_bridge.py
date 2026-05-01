"""Runtime skill context bridge interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any

from Tool.BaseTool import ToolResult
from core.cache_policy import CacheableBlock
from core.Message import MetaUserMessage
from logging import getLogger
logger = getLogger(__name__)
from agent import BasicAgent
from core.request_input import ReplayRequestInput
class BaseRuntimeSkillContextBridge(ABC):
    """Bridge for runtime skill/tool ephemeral context injection."""

    @abstractmethod
    def append_runtime_skill_context_message(self, agent:BasicAgent, messages: list[Any]) -> None:
        pass

    @abstractmethod
    def append_tool_ephemeral_context_message(
        self,
        agent:BasicAgent,
        *,
        tool_name: str,
        context: Any,
        messages: list[Any],
    ) -> None:
        pass

    @abstractmethod
    def clear_ephemeral_skill_state(self, agent:BasicAgent) -> None:
        pass

    @abstractmethod
    def maybe_inject_runtime_skill_context(
        self,
        agent:BasicAgent,
        *,
        tool_name: str,
        messages: list[Any],
    ) -> None:
        pass

    @abstractmethod
    def maybe_inject_tool_ephemeral_context(
        self,
        agent:BasicAgent,
        *,
        tool_name: str,
        tool_result: ToolResult,
        messages: list[Any],
    ) -> None:
        pass


class DefaultRuntimeSkillContextBridge(BaseRuntimeSkillContextBridge):
    """Default runtime context bridge that preserves current BasicAgent behavior."""

    @staticmethod
    def _append_runtime_text(agent:BasicAgent, messages: list[Any] | ReplayRequestInput, text: str) -> None:
        if isinstance(messages, ReplayRequestInput):
            messages.append_dynamic_tail_text(text)
            return
        messages.append(MetaUserMessage(text, metadata={"source": "skill_tool"}))

    @staticmethod
    def _append_runtime_skill_blocks(agent: BasicAgent, messages: list[Any] | ReplayRequestInput) -> bool:
        if not isinstance(messages, ReplayRequestInput):
            return False
        blocks = agent.skill_manager.build_runtime_skill_context_blocks()
        if not blocks:
            return False
        for block in blocks:
            messages.append_on_demand_expansion_block(block)
        return True

    def append_runtime_skill_context_message(self, agent:BasicAgent, messages: list[Any]) -> None:
        runtime_prompt = agent.skill_manager.build_runtime_skill_context_prompt()
        if self._append_runtime_skill_blocks(agent, messages):
            logger.info("Injecting runtime skill context as on-demand expansion")
            return
        if not runtime_prompt:
            logger.debug("No runtime skill context available")
            return
        logger.info("Injecting runtime skill context")
        self._append_runtime_text(agent, messages, runtime_prompt)

    def append_tool_ephemeral_context_message(
        self,
        agent:BasicAgent,
        *,
        tool_name: str,
        context: Any,
        messages: list[Any],
    ) -> None:
        if context is None:
            return
        if isinstance(context, str):
            context_text = context.strip()
        elif isinstance(context, (dict, list)):
            context_text = json.dumps(context, ensure_ascii=False, indent=2)
        else:
            context_text = str(context).strip()
        if not context_text:
            return
        logger.info("Injecting tool ephemeral context")
        if isinstance(messages, ReplayRequestInput):
            messages.append_dynamic_tail_block(
                CacheableBlock(
                    name=f"runtime_tool_context:{tool_name}",
                    content=f"## Runtime Tool Context\n<runtime-tool-context tool=\"{tool_name}\">\n{context_text}\n</runtime-tool-context>",
                    partition="dynamic",
                    cacheable=False,
                    reason="runtime_tool_context",
                    metadata={"tool_name": tool_name},
                )
            )
            return
        self._append_runtime_text(agent, messages, f"## Runtime Tool Context\n<runtime-tool-context tool=\"{tool_name}\">\n{context_text}\n</runtime-tool-context>")

    def clear_ephemeral_skill_state(self, agent:BasicAgent) -> None:
        agent.skill_manager.clear_ephemeral_state()

    def maybe_inject_runtime_skill_context(
        self,
        agent:BasicAgent,
        *,
        tool_name: str,
        messages: list[Any],
    ) -> None:
        if tool_name != "skill_tool":
            return
        if not agent.skill_manager.has_runtime_skill_context():
            return
        self.append_runtime_skill_context_message(agent, messages)

    def maybe_inject_tool_ephemeral_context(
        self,
        agent:BasicAgent,
        *,
        tool_name: str,
        tool_result: ToolResult,
        messages: list[Any],
    ) -> None:
        self.append_tool_ephemeral_context_message(
            agent,
            tool_name=tool_name,
            context=tool_result.ephemeral_context,
            messages=messages,
        )
