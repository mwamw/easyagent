from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

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
    runtime_reminder_blocks: list[CacheableBlock]
    dynamic_tail_blocks: list[CacheableBlock]
    on_demand_expansion_blocks: list[CacheableBlock]
    cache_policy: PromptCachePolicy
    metadata: dict[str, Any]

    @property
    def dynamic_context_blocks(self) -> list[CacheableBlock]:
        return self.dynamic_tail_blocks


def _force_dynamic(block: CacheableBlock, *, reason: str) -> CacheableBlock:
    return CacheableBlock(
        name=block.name,
        content=block.content,
        partition="dynamic",
        cacheable=False,
        reason=reason,
        metadata=dict(block.metadata or {}),
    )


def compile_prompt_blocks(
    blocks: Iterable[Any],
    *,
    cache_policy: PromptCachePolicy | dict[str, Any] | None = None,
    cache_dynamic_memory: bool = False,
    cache_dynamic_mailbox: bool = False,
    cache_turn_skills: bool = False,
) -> CompiledPrompt:
    """Split prompt blocks into cacheable system prefix and dynamic context.

    The default policy keeps volatile blocks out of the cacheable system prefix.
    Existing prompt blocks can override this with `metadata["cache_partition"]`.
    """

    policy = normalize_cache_policy(cache_policy)
    system_blocks: list[CacheableBlock] = []
    reminder_blocks: list[CacheableBlock] = []
    dynamic_blocks: list[CacheableBlock] = []
    expansion_blocks: list[CacheableBlock] = []

    for raw_block in blocks:
        block = prompt_block_to_cacheable(raw_block)
        name = block.name
        metadata = dict(block.metadata or {})
        request_layer = str(metadata.get("request_layer") or "").strip().lower()

        if name == "memory" and not cache_dynamic_memory:
            block = _force_dynamic(block, reason="memory_is_dynamic")
        elif name == "mailbox" and not cache_dynamic_mailbox:
            block = _force_dynamic(block, reason="mailbox_is_dynamic")
        elif metadata.get("skill_lifecycle") == "turn" and not cache_turn_skills:
            block = _force_dynamic(block, reason="turn_skill_is_dynamic")
        elif block.partition == "dynamic":
            block = _force_dynamic(block, reason=block.reason or "marked_dynamic")

        if request_layer == "on_demand_expansion":
            expansion_blocks.append(block)
        elif request_layer == "reminder" or name in {"tool_inventory", "skill_listing"}:
            reminder_blocks.append(block)
        elif block.partition == "dynamic":
            dynamic_blocks.append(block)
        else:
            system_blocks.append(block)

    return CompiledPrompt(
        system_prompt=render_blocks(system_blocks, include_dynamic=False),
        system_prompt_blocks=system_blocks,
        runtime_reminder_blocks=reminder_blocks,
        dynamic_tail_blocks=dynamic_blocks,
        on_demand_expansion_blocks=expansion_blocks,
        cache_policy=policy,
        metadata={
            "systemBlockCount": len(system_blocks),
            "runtimeReminderBlockCount": len(reminder_blocks),
            "dynamicBlockCount": len(dynamic_blocks),
            "onDemandExpansionBlockCount": len(expansion_blocks),
            "runtimeReminderNames": [block.name for block in reminder_blocks],
            "dynamicBlockNames": [block.name for block in dynamic_blocks],
            "onDemandExpansionNames": [block.name for block in expansion_blocks],
        },
    )


__all__ = ["CompiledPrompt", "compile_prompt_blocks"]
