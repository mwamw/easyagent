"""Public extension contract for optional multi-agent modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from prompt import PromptBlock, PromptBuildContext
    from runtime import AgentRuntime, RuntimeEventBus
    from runtime.metamessage import MetaMessageManager
    from Tool.BaseTool import Tool
    from Tool.ToolRegistry import ToolRegistry


class BaseMultiAgent(ABC):
    """Integration boundary for a replaceable multi-agent implementation.

    The framework owns composition. Implementations expose their tools and
    prompt contributions, receive only the shared facilities they actually
    need, and keep their own orchestration model private.
    """

    @abstractmethod
    def bind(
        self,
        *,
        tool_registry: "ToolRegistry",
        runtime: "AgentRuntime",
    ) -> None:
        """Bind framework facilities without receiving the parent Agent."""

    @abstractmethod
    def get_tools(self) -> Sequence["Tool"]:
        """Return model-facing tools contributed by this implementation."""

    def get_prompt_blocks(self, context: "PromptBuildContext") -> Sequence["PromptBlock"]:
        """Return request-time prompt guidance for this implementation."""

        return ()

    def before_context_build(self) -> None:
        """Refresh dynamic collaboration input before context is assembled."""

    def export_state(self) -> dict[str, Any]:
        """Return JSON-serializable session state."""

        return {}

    def restore_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Restore state and return a structured restore report."""

        return {
            "status": "restored",
            "restoredItems": [],
            "degradedItems": [],
            "missingItems": [],
            "metadata": {},
            "issues": [],
        }

    def close(self) -> dict[str, Any]:
        """Release implementation-owned resources."""

        return {"status": "closed", "issues": []}


__all__ = ["BaseMultiAgent"]
