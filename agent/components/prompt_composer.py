"""Prompt composer interfaces and default implementations for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional

from prompt import (
    PromptBlock,
    SystemPromptTemplate,
    build_output_efficiency_section,
    build_safety_section,
    build_task_execution_section,
    build_tone_style_section,
    build_tool_policy_section,
    build_visibility_section,
    format_tool_catalog,
    format_tool_inventory,
)

logger = logging.getLogger(__name__)


class BasePromptComposer(ABC):
    """Abstract prompt composer used by BasicAgent."""

    @abstractmethod
    def get_enhanced_prompt(self, agent: Any) -> str:
        pass

    @abstractmethod
    def get_system_prompt_template(self, agent: Any) -> SystemPromptTemplate:
        pass

    @abstractmethod
    def get_system_prompt_blocks(self, agent: Any) -> list[PromptBlock]:
        pass

    @abstractmethod
    def build_core_prompt_blocks(
        self,
        agent: Any,
        *,
        start_order: int,
        include_tool_policy: bool,
    ) -> list[PromptBlock]:
        pass

    @abstractmethod
    def get_tool_catalog_prompt(self, agent: Any) -> str:
        pass

    @abstractmethod
    def get_tool_inventory_prompt(self, agent: Any, *, include_parameters: bool = False) -> str:
        pass

    @abstractmethod
    def build_tool_inventory_block(self, agent: Any, order: int) -> PromptBlock | None:
        pass

    @abstractmethod
    def build_shared_prompt_blocks(
        self,
        agent: Any,
        *,
        start_order: int,
        include_custom_prompt: bool = True,
        include_memory: bool = True,
        include_skills: bool = True,
    ) -> list[PromptBlock]:
        pass

    @abstractmethod
    def get_extension_prompt_blocks(self, start_order: int) -> list[PromptBlock]:
        pass

    @abstractmethod
    def with_prompt_block(self, block: PromptBlock) -> None:
        pass

    @abstractmethod
    def with_prompt_blocks(self, blocks: list[PromptBlock]) -> None:
        pass


class DefaultPromptComposer(BasePromptComposer):
    """Default prompt composer that preserves current BasicAgent behavior."""

    def __init__(self):
        self._extension_prompt_blocks: list[PromptBlock] = []

    def get_enhanced_prompt(self, agent: Any) -> str:
        return self.get_system_prompt_template(agent).render()

    def get_system_prompt_template(self, agent: Any) -> SystemPromptTemplate:
        return SystemPromptTemplate(agent.get_system_prompt_blocks())

    def get_system_prompt_blocks(self, agent: Any) -> list[PromptBlock]:
        if not agent.enable_tool or not agent.tool_registry:
            blocks = [
                PromptBlock(
                    name="identity",
                    content=agent.system_prompt or "你是一个有用的 AI 助手，帮助用户回答问题并完成任务。",
                    order=0,
                )
            ]
            blocks.extend(self.build_core_prompt_blocks(agent, start_order=10, include_tool_policy=False))
            blocks.extend(
                self.build_shared_prompt_blocks(
                    agent,
                    start_order=100,
                    include_custom_prompt=False,
                )
            )
            return blocks

        blocks = [
            PromptBlock(
                name="identity",
                content="你是一个智能助手，具备使用工具解决问题的能力。",
                order=0,
            ),
        ]
        blocks.extend(self.build_core_prompt_blocks(agent, start_order=10, include_tool_policy=True))
        tool_block = self.build_tool_inventory_block(agent, order=60)
        if tool_block is not None:
            blocks.append(tool_block)
        blocks.extend(self.build_shared_prompt_blocks(agent, start_order=100))
        return blocks

    def build_core_prompt_blocks(
        self,
        agent: Any,
        *,
        start_order: int,
        include_tool_policy: bool,
    ) -> list[PromptBlock]:
        blocks = [
            PromptBlock(name="visibility", content=build_visibility_section(), order=start_order),
            PromptBlock(name="task_execution", content=build_task_execution_section(), order=start_order + 10),
            PromptBlock(name="safety", content=build_safety_section(), order=start_order + 20),
        ]
        if include_tool_policy:
            blocks.append(
                PromptBlock(
                    name="tool_policy",
                    content=build_tool_policy_section(),
                    order=start_order + 30,
                )
            )
            style_order = start_order + 40
        else:
            style_order = start_order + 30

        blocks.extend(
            [
                PromptBlock(name="tone_style", content=build_tone_style_section(), order=style_order),
                PromptBlock(
                    name="output_efficiency",
                    content=build_output_efficiency_section(),
                    order=style_order + 10,
                ),
            ]
        )
        return blocks

    def get_tool_catalog_prompt(self, agent: Any) -> str:
        if not agent.tool_registry:
            return ""
        try:
            tool_descriptions = agent.tool_registry.get_tools_description()
        except Exception as exc:
            logger.error(f"获取工具描述失败: {exc}")
            return "（工具描述获取失败）"
        return format_tool_catalog(tool_descriptions)

    def get_tool_inventory_prompt(self, agent: Any, *, include_parameters: bool = False) -> str:
        if not agent.tool_registry:
            return ""
        try:
            tool_descriptions = agent.tool_registry.get_tools_description()
        except Exception as exc:
            logger.error(f"获取工具描述失败: {exc}")
            return "（工具描述获取失败）"
        return format_tool_inventory(tool_descriptions, include_parameters=include_parameters)

    def build_tool_inventory_block(self, agent: Any, order: int) -> PromptBlock | None:
        if not agent._should_include_tool_inventory_block():
            return None

        mode = agent._tool_inventory_mode()
        if mode == "none":
            return None

        content = self.get_tool_inventory_prompt(agent, include_parameters=(mode == "full"))
        if not content:
            return None

        title = "## 可用工具"
        if mode == "compact":
            title = "## 可用工具概览"

        return PromptBlock(
            name="tool_inventory",
            content=f"{title}\n{content}",
            order=order,
        )

    def build_shared_prompt_blocks(
        self,
        agent: Any,
        *,
        start_order: int,
        include_custom_prompt: bool = True,
        include_memory: bool = True,
        include_skills: bool = True,
    ) -> list[PromptBlock]:
        blocks: list[PromptBlock] = []
        order = start_order

        if include_custom_prompt and agent.system_prompt and agent.system_prompt.strip():
            blocks.append(
                PromptBlock(
                    name="custom_instructions",
                    content=f"## 额外指令\n{agent.system_prompt.strip()}",
                    order=order,
                )
            )
            order += 10

        skill_policy_prompt = agent.skill_manager.build_skill_policy_prompt()
        if skill_policy_prompt:
            blocks.append(
                PromptBlock(
                    name="skill_policy",
                    content=skill_policy_prompt,
                    order=order,
                )
            )
            order += 10

        skill_listing_prompt = agent.skill_manager.build_skill_listing_prompt()
        if skill_listing_prompt:
            blocks.append(
                PromptBlock(
                    name="skill_listing",
                    content=skill_listing_prompt,
                    order=order,
                )
            )
            order += 10

        memory_prompt = agent._build_memory_prompt() if include_memory else ""
        if memory_prompt:
            blocks.append(
                PromptBlock(
                    name="memory",
                    content=memory_prompt,
                    order=order,
                )
            )
            order += 10

        exclude_names = {"memory"} if memory_prompt else None
        if include_skills:
            skills_prompt = agent._build_skills_prompt(exclude_names=exclude_names)
            if skills_prompt:
                blocks.append(
                    PromptBlock(
                        name="skills",
                        content=skills_prompt,
                        order=order,
                    )
                )
                order += 10

        blocks.extend(self.get_extension_prompt_blocks(start_order=order))
        return blocks

    def get_extension_prompt_blocks(self, start_order: int) -> list[PromptBlock]:
        blocks = list(self._extension_prompt_blocks)
        normalized: list[PromptBlock] = []
        for index, block in enumerate(blocks):
            normalized.append(
                PromptBlock(
                    name=block.name,
                    content=block.content,
                    order=start_order + index * 10 if block.order == 0 else block.order,
                    enabled=block.enabled,
                    metadata=dict(block.metadata),
                )
            )
        return normalized

    def with_prompt_block(self, block: PromptBlock) -> None:
        self._extension_prompt_blocks.append(block)

    def with_prompt_blocks(self, blocks: list[PromptBlock]) -> None:
        self._extension_prompt_blocks.extend(blocks)
