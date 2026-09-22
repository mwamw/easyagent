"""Public prompt-composition contract and contribution context."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from core.Config import Config
from runtime import AgentRuntime
from .system_prompt import PromptBlock

def merge_prompt_blocks(blocks: Iterable[PromptBlock]) -> list[PromptBlock]:
    merged: list[PromptBlock] = []
    indexes: dict[str, int] = {}
    for block in blocks:
        if not isinstance(block, PromptBlock):
            raise TypeError(f"block must be PromptBlock, got {type(block).__name__}")
        index = indexes.get(block.name)
        if index is None:
            indexes[block.name] = len(merged)
            merged.append(block)
        else:
            merged[index] = block
    return merged



@dataclass(slots=True, frozen=True)
class PromptBuildContext:
    """Public request-time data available to custom prompt composers."""

    agent_name: str
    description: str | None
    system_prompt: str | None
    query: str
    config: Config
    runtime: AgentRuntime
    tools_enabled: bool = False
    providers: tuple[Callable[["PromptBuildContext"], Iterable[PromptBlock]], ...] = ()


class BaseSystemPromptComposer(ABC):
    """Module extension boundary; subclasses normally override only ``build``."""

    include_module_blocks = True

    @abstractmethod
    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return []

    def compose(self, context: PromptBuildContext) -> list[PromptBlock]:
        module_blocks = [
            block for provider in context.providers for block in provider(context)
        ] if self.include_module_blocks else []
        blocks = merge_prompt_blocks([*module_blocks, *self.build(context)])
        for block in blocks:
            if not isinstance(block, PromptBlock):
                raise TypeError(
                    "System prompt composers must return PromptBlock values, "
                    f"got {type(block).__name__}"
                )
        return [block.resolve(context) for block in blocks if block.enabled]
