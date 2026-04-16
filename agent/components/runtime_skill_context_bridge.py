"""Runtime skill context bridge interfaces and default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any

from Tool.BaseTool import ToolResult
from core.Message import MetaUserMessage


class BaseRuntimeSkillContextBridge(ABC):
    """Bridge for runtime skill/tool ephemeral context injection."""

    @abstractmethod
    def append_runtime_skill_context_message(self, agent: Any, messages: list[Any]) -> None:
        pass

    @abstractmethod
    def append_tool_ephemeral_context_message(
        self,
        agent: Any,
        *,
        tool_name: str,
        context: Any,
        messages: list[Any],
    ) -> None:
        pass

    @abstractmethod
    def clear_ephemeral_skill_state(self, agent: Any) -> None:
        pass

    @abstractmethod
    def maybe_inject_runtime_skill_context(
        self,
        agent: Any,
        *,
        tool_name: str,
        messages: list[Any],
    ) -> None:
        pass

    @abstractmethod
    def maybe_inject_tool_ephemeral_context(
        self,
        agent: Any,
        *,
        tool_name: str,
        tool_result: ToolResult,
        messages: list[Any],
    ) -> None:
        pass


class DefaultRuntimeSkillContextBridge(BaseRuntimeSkillContextBridge):
    """Default runtime context bridge that preserves current BasicAgent behavior."""

    def append_runtime_skill_context_message(self, agent: Any, messages: list[Any]) -> None:
        runtime_prompt = agent.skill_manager.build_runtime_skill_context_prompt()
        if not runtime_prompt:
            return
        messages.append(
            MetaUserMessage(
                runtime_prompt,
                metadata={
                    "source": "skill_tool",
                    "kind": "runtime_skill_context",
                },
            )
        )

    def append_tool_ephemeral_context_message(
        self,
        agent: Any,
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
        messages.append(
            MetaUserMessage(
                f"## Runtime Tool Context\n<runtime-tool-context tool=\"{tool_name}\">\n{context_text}\n</runtime-tool-context>",
                metadata={
                    "source": tool_name,
                    "kind": "runtime_tool_context",
                },
            )
        )

    def clear_ephemeral_skill_state(self, agent: Any) -> None:
        agent.skill_manager.clear_ephemeral_state()

    def maybe_inject_runtime_skill_context(
        self,
        agent: Any,
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
        agent: Any,
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
