from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from prompt import SystemPromptTemplate

from .cache_policy import (
    CacheableBlock,
    PromptCachePolicy,
    normalize_cache_policy,
    prompt_block_to_cacheable,
    render_blocks,
)


@dataclass(slots=True)
class CompiledPrompt:
    system_prompt: Optional[str]
    system_prompt_blocks: list[CacheableBlock]
    system_reminder_blocks: list[CacheableBlock]
    dynamic_tail_blocks: list[CacheableBlock]
    on_demand_expansion_blocks: list[CacheableBlock]
    cache_policy: PromptCachePolicy
    metadata: dict[str, Any]

    @property
    def dynamic_context_blocks(self) -> list[CacheableBlock]:
        return self.dynamic_tail_blocks


def compile_prompt_blocks(
    blocks: Iterable[Any],
    *,
    cache_policy: PromptCachePolicy | dict[str, Any] | None = None,
) -> CompiledPrompt:
    """Compile unified prompt blocks according to their placement field."""

    policy = normalize_cache_policy(cache_policy)
    system_blocks: list[CacheableBlock] = []
    reminder_blocks: list[CacheableBlock] = []
    dynamic_blocks: list[CacheableBlock] = []
    expansion_blocks: list[CacheableBlock] = []

    template = SystemPromptTemplate(blocks)
    for raw_block in template.get_blocks(placement="system"):
        block = prompt_block_to_cacheable(raw_block)
        system_blocks.append(block)
    for raw_block in template.get_blocks(placement="system_reminder"):
        reminder_blocks.append(prompt_block_to_cacheable(raw_block))

    return CompiledPrompt(
        system_prompt=render_blocks(system_blocks, include_dynamic=True),
        system_prompt_blocks=system_blocks,
        system_reminder_blocks=reminder_blocks,
        dynamic_tail_blocks=dynamic_blocks,
        on_demand_expansion_blocks=expansion_blocks,
        cache_policy=policy,
        metadata={
            "systemBlockCount": len(system_blocks),
            "systemReminderBlockCount": len(reminder_blocks),
            "dynamicBlockCount": len(dynamic_blocks),
            "onDemandExpansionBlockCount": len(expansion_blocks),
            "systemReminderNames": [block.name for block in reminder_blocks],
            "dynamicBlockNames": [block.name for block in dynamic_blocks],
            "onDemandExpansionNames": [block.name for block in expansion_blocks],
        },
    )


__all__ = ["CompiledPrompt", "compile_prompt_blocks"]
