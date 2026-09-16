"""System-prompt composition and registered module contributions."""

from __future__ import annotations

from typing import Any, Iterable

from prompt import (
    PromptBlock,
    SystemPromptTemplate,
    build_output_efficiency_section,
    build_safety_section,
    build_task_execution_section,
    build_tone_style_section,
    build_tool_policy_section,
    build_visibility_section,
)

from .base import BaseSystemPromptComposer, PromptBuildContext, merge_prompt_blocks


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

    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        defaults = self._default_blocks(context) if self.include_defaults else []
        return merge_prompt_blocks([*defaults, *self._blocks])

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
        return blocks

    def build_capability_blocks(self, context: PromptBuildContext, *, language="zh") -> list[PromptBlock]:
        """Collect contributions without depending on any concrete module."""
        return [block for provider in context.providers for block in provider(context)]

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
