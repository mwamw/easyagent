"""System-prompt composition with a small public customization surface."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import logging
from typing import Any, Iterable

from core.Config import Config
from prompt import (
    PromptBlock,
    SystemPromptTemplate,
    build_output_efficiency_section,
    build_safety_section,
    build_task_execution_section,
    build_tone_style_section,
    build_tool_policy_section,
    build_visibility_section,
    format_tool_inventory,
)
from runtime import ExecutionContext
from skill.manager import SkillManager
from Tool.ToolRegistry import ToolRegistry


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PromptBuildContext:
    """Public request-time data available to custom prompt composers."""

    agent_name: str
    description: str | None
    system_prompt: str | None
    query: str
    config: Config
    execution_context: ExecutionContext
    tool_registry: ToolRegistry | None
    skill_manager: SkillManager | None
    memory: Any = None
    task_service: Any = None
    plan: Any = None
    context_manager: Any = None

    @property
    def tools_enabled(self) -> bool:
        return self.tool_registry is not None and bool(self.tool_registry.get_tool_names())


class BaseSystemPromptComposer(ABC):
    """Module extension boundary; subclasses normally override only ``build``."""

    include_module_blocks = True

    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return []

    def compose(self, context: PromptBuildContext) -> list[PromptBlock]:
        module_blocks = (
            context.memory.get_prompt_blocks(context)
            if self.include_module_blocks and context.memory is not None else []
        )
        blocks = SystemPromptComposer._merge([*module_blocks, *self.build(context)])
        for block in blocks:
            if not isinstance(block, PromptBlock):
                raise TypeError(
                    "System prompt composers must return PromptBlock values, "
                    f"got {type(block).__name__}"
                )
        return [block.resolve(context) for block in blocks if block.enabled]


class SystemPromptComposer(BaseSystemPromptComposer):
    """Industrial default prompt composer plus optional user-defined blocks."""

    def __init__(
        self,
        blocks: Iterable[PromptBlock] | None = None,
        *,
        include_defaults: bool = True,
        include_module_blocks: bool = True,
    ) -> None:
        self.include_defaults = bool(include_defaults)
        self.include_module_blocks = bool(include_module_blocks)
        self._blocks = list(blocks or [])
        for block in self._blocks:
            if not isinstance(block, PromptBlock):
                raise TypeError(f"block must be PromptBlock, got {type(block).__name__}")

    def add_block(self, block: PromptBlock) -> "SystemPromptComposer":
        if not isinstance(block, PromptBlock):
            raise TypeError(f"block must be PromptBlock, got {type(block).__name__}")
        self._blocks.append(block)
        return self

    @staticmethod
    def _merge(blocks: Iterable[PromptBlock]) -> list[PromptBlock]:
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

    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        defaults = self._default_blocks(context) if self.include_defaults else []
        return self._merge([*defaults, *self._blocks])

    def _default_blocks(self, context: PromptBuildContext) -> list[PromptBlock]:
        identity = context.system_prompt or (
            "你是一个智能助手，可以使用当前请求提供的工具完成任务。"
            if context.tools_enabled
            else "你是一个有用的 AI 助手，帮助用户回答问题并完成任务。"
        )
        static = {"cache_partition": "static", "cacheable": True}
        blocks = [
            PromptBlock("identity", identity, order=0, metadata={"cache_partition": "session", "cacheable": True}),
            PromptBlock("visibility", build_visibility_section(), order=10, metadata=static),
            PromptBlock("task_execution", build_task_execution_section(), order=20, metadata=static),
            PromptBlock("safety", build_safety_section(), order=30, metadata=static),
        ]
        next_order = 40
        if context.tools_enabled:
            blocks.append(PromptBlock("tool_policy", build_tool_policy_section(), order=next_order, metadata=static))
            next_order += 10
        blocks.extend(
            [
                PromptBlock("tone_style", build_tone_style_section(), order=next_order, metadata=static),
                PromptBlock("output_efficiency", build_output_efficiency_section(), order=next_order + 10, metadata=static),
            ]
        )
        blocks.extend(self.build_capability_blocks(context))
        return blocks

    def build_capability_blocks(
        self, context: PromptBuildContext, *, language: str = "zh"
    ) -> list[PromptBlock]:
        """Describe installed capabilities without adding default behavior rules.

        Products owning their entire system prompt can reuse capability discovery
        with ``include_defaults=False`` instead of copying framework composition.
        """
        blocks = self._skill_blocks(context, start_order=100)
        inventory = self._tool_inventory_block(context, order=160)
        if inventory is not None:
            blocks.append(inventory)
        if context.plan is not None and context.plan.mode == "plan":
            blocks.append(PromptBlock(
                "plan_mode",
                "Plan mode still active (see full instructions earlier in conversation). "
                "Read-only inspection and planning only; do not modify files or run non-readonly tools. "
                "Follow the 5-phase workflow: understand, design, review, present the plan, request approval. "
                "Use AskUserQuestion for clarifications or ExitPlanMode for plan approval when available. "
                "Never ask about plan approval via text or AskUserQuestion.",
                placement="system_reminder", order=190,
                metadata={"cache_partition": "dynamic", "cacheable": False},
            ))
        if context.context_manager is not None:
            blocks.append(PromptBlock(
                "compaction",
                "Auto-compact is enabled. When the context window is nearly full, older messages "
                "will be automatically compacted so you can continue working. Continue from the "
                "retained context; do not stop or rush merely because the conversation is long. "
                "Re-read source files when exact details are no longer available after compaction.",
                placement="system_reminder", order=200,
            ))
        return blocks

    @staticmethod
    def _skill_blocks(context: PromptBuildContext, *, start_order: int) -> list[PromptBlock]:
        if context.skill_manager is None:
            return []
        listing = context.skill_manager.build_skill_listing_prompt()
        if listing:
            return [
                PromptBlock(
                    "skill_listing",
                    listing,
                    placement="system_reminder",
                    order=start_order,
                )
            ]
        return []

    @staticmethod
    def _tool_inventory_block(context: PromptBuildContext, *, order: int) -> PromptBlock | None:
        registry = context.tool_registry
        if registry is None or context.config.tool_schema_mode != "deferred":
            return None
        try:
            descriptors = registry.list_tool_descriptors(stable=True, include_parameters=False)
        except Exception as exc:
            logger.warning("Failed to build deferred tool inventory: %s", exc)
            return None
        content = format_tool_inventory(descriptors, include_parameters=False)
        return PromptBlock(
            "tool_inventory",
            "## Available tools\nTool schemas are loaded on demand. Call `tool_schema_tool` to load the schemas of tools not yet exposed in this request.\n\n" + content,
            placement="system_reminder",
            order=order,
            metadata={"cache_partition": "session", "cacheable": True},
        )

    def get_system_prompt_template(self, context: PromptBuildContext) -> SystemPromptTemplate:
        return SystemPromptTemplate(self.compose(context))

    def get_enhanced_prompt(self, context: PromptBuildContext) -> str:
        return self.get_system_prompt_template(context).render_system()

    def export_state(self) -> dict[str, Any]:
        serialized: list[dict[str, Any]] = []
        for block in self._blocks:
            if callable(block.content):
                continue
            serialized.append(
                {
                    "name": block.name,
                    "content": block.content,
                    "placement": block.placement,
                    "order": block.order,
                    "enabled": block.enabled,
                    "metadata": dict(block.metadata),
                }
            )
        return {"includeDefaults": self.include_defaults, "blocks": serialized}

    def restore_state(self, state: dict[str, Any] | None) -> None:
        payload = dict(state or {})
        self.include_defaults = bool(payload.get("includeDefaults", True))
        self._blocks = [PromptBlock(**dict(item)) for item in list(payload.get("blocks") or [])]


__all__ = ["BaseSystemPromptComposer", "PromptBuildContext", "SystemPromptComposer"]
