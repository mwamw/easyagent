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
from agent import BasicAgent
logger = logging.getLogger(__name__)


class BasePromptComposer(ABC):
    """Abstract prompt composer used by BasicAgent.

    负责将 agent 的各类配置（身份、工具、技能、记忆等）组装为最终的
    系统提示词。通过实现不同的子类可以完全自定义提示词的构建逻辑。
    """

    @abstractmethod
    def get_enhanced_prompt(self, agent:BasicAgent) -> str:
        """组装并返回最终渲染后的系统提示词纯文本。

        这是最高层的入口方法，内部会调用 get_system_prompt_template()
        获取模板对象，再调用 render() 将所有 PromptBlock 按 order 排序
        拼接为一个完整的字符串，直接用于 LLM 请求的 system prompt。
        """

    @abstractmethod
    def get_system_prompt_template(self, agent:BasicAgent) -> SystemPromptTemplate:
        """构建并返回 SystemPromptTemplate 对象。

        SystemPromptTemplate 持有一组 PromptBlock，提供 render() 方法
        将它们按 order 排序后拼接。此方法是 get_enhanced_prompt() 与
        需要直接操作模板对象的调用方之间的桥梁。
        """

    @abstractmethod
    def get_system_prompt_blocks(self, agent:BasicAgent) -> list[PromptBlock]:
        """构建完整的系统提示词块列表。

        这是提示词组装的核心编排方法。根据 agent 是否启用工具，决定：
        - 使用哪个 identity block（纯文本助手 vs 工具助手）
        - 是否包含工具策略和工具清单
        - 如何组合 core blocks 和 shared blocks
        返回的列表会被 SystemPromptTemplate 消费。
        """

    @abstractmethod
    def build_core_prompt_blocks(
        self,
        agent:BasicAgent,
        *,
        start_order: int,
        include_tool_policy: bool,
    ) -> list[PromptBlock]:
        """构建核心行为准则相关的 prompt blocks。

        包含与 agent 具体配置无关的通用行为指引：
        - visibility: 可见性/上下文感知规则
        - task_execution: 任务执行策略
        - safety: 安全边界约束
        - tool_policy: 工具使用策略（仅当 include_tool_policy=True 时）
        - tone_style: 语气与风格
        - output_efficiency: 输出效率要求

        Args:
            start_order: 第一个 block 的 order 值，后续 block 递增。
            include_tool_policy: 是否包含工具使用策略 block。
        """

    @abstractmethod
    def get_tool_catalog_prompt(self, agent:BasicAgent) -> str:
        """返回详细的工具目录提示词文本。

        从 agent.tool_registry 获取所有工具描述，使用 format_tool_catalog()
        格式化为包含工具名称、描述、参数等完整信息的文本。
        适用于需要向 LLM 展示工具详细规格的场景。
        """

    @abstractmethod
    def get_tool_inventory_prompt(self, agent:BasicAgent, *, include_parameters: bool = False) -> str:
        """返回工具清单提示词文本。

        与 get_tool_catalog_prompt() 类似，但使用 format_tool_inventory()
        格式化，支持通过 include_parameters 控制是否包含参数详情：
        - include_parameters=False: 仅列出工具名称和简要描述（compact 模式）
        - include_parameters=True: 包含完整参数信息（full 模式）
        """

    @abstractmethod
    def build_tool_inventory_block(self, agent:BasicAgent, order: int) -> PromptBlock | None:
        """构建工具清单 PromptBlock，若不需要则返回 None。

        根据 agent 的配置（_should_include_tool_inventory_block 和
        _tool_inventory_mode）决定是否生成工具清单 block 以及使用
        compact 还是 full 模式。返回 None 表示不在系统提示词中嵌入
        工具清单。

        Args:
            order: 该 block 在最终提示词中的排序位置。
        """

    @abstractmethod
    def build_shared_prompt_blocks(
        self,
        agent:BasicAgent,
        *,
        start_order: int,
        include_custom_prompt: bool = True,
        include_memory: bool = True,
        include_skills: bool = True,
    ) -> list[PromptBlock]:
        """构建共享/可选的 prompt blocks。

        包含与 agent 实例特定配置相关的内容：
        - custom_instructions: 用户自定义的额外系统提示词
        - skill_policy: 技能使用策略
        - skill_listing: 可用技能列表
        - memory: agent 的持久化记忆内容
        - mailbox: 来自其他 agent 的消息（A2A 场景）
        - skills: 技能详情
        - extension blocks: 通过 with_prompt_block(s) 注入的扩展块

        各参数控制是否包含对应部分，允许不同场景下灵活裁剪。

        Args:
            start_order: 起始排序值，各 block 按出现顺序递增。
            include_custom_prompt: 是否包含用户自定义指令。
            include_memory: 是否包含记忆内容。
            include_skills: 是否包含技能详情。
        """

    @abstractmethod
    def get_extension_prompt_blocks(self, start_order: int) -> list[PromptBlock]:
        """返回通过 with_prompt_block(s) 注册的扩展 prompt blocks。

        扩展 blocks 允许外部代码在不修改 composer 核心逻辑的情况下
        向系统提示词注入额外内容（如插件说明、环境上下文等）。
        返回时会对 order=0 的 block 重新分配排序值。

        Args:
            start_order: 未指定 order 的 block 从此值开始分配。
        """

    @abstractmethod
    def with_prompt_block(self, block: PromptBlock) -> None:
        """注册单个扩展 prompt block。

        注册的 block 会在 build_shared_prompt_blocks() 末尾通过
        get_extension_prompt_blocks() 被收集并追加到最终提示词中。
        """

    @abstractmethod
    def with_prompt_blocks(self, blocks: list[PromptBlock]) -> None:
        """批量注册多个扩展 prompt blocks。

        等价于对每个 block 依次调用 with_prompt_block()。
        """


class DefaultPromptComposer(BasePromptComposer):
    """Default prompt composer that preserves current BasicAgent behavior."""

    def __init__(self):
        self._extension_prompt_blocks: list[PromptBlock] = []

    def get_enhanced_prompt(self, agent:BasicAgent) -> str:
        return self.get_system_prompt_template(agent).render()

    def get_system_prompt_template(self, agent:BasicAgent) -> SystemPromptTemplate:
        return SystemPromptTemplate(self.get_system_prompt_blocks(agent))

    def get_system_prompt_blocks(self, agent:BasicAgent) -> list[PromptBlock]:
        if not agent.enable_tool or not agent.tool_registry:
            blocks = [
                PromptBlock(
                    name="identity",
                    content=agent.system_prompt or "你是一个有用的 AI 助手，帮助用户回答问题并完成任务。",
                    order=0,
                    metadata={"cache_partition": "static", "cacheable": True},
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
                metadata={"cache_partition": "static", "cacheable": True},
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
        agent:BasicAgent,
        *,
        start_order: int,
        include_tool_policy: bool,
    ) -> list[PromptBlock]:
        blocks = [
            PromptBlock(name="visibility", content=build_visibility_section(), order=start_order, metadata={"cache_partition": "static", "cacheable": True}),
            PromptBlock(name="task_execution", content=build_task_execution_section(), order=start_order + 10, metadata={"cache_partition": "static", "cacheable": True}),
            PromptBlock(name="safety", content=build_safety_section(), order=start_order + 20, metadata={"cache_partition": "static", "cacheable": True}),
        ]
        if include_tool_policy:
            blocks.append(
                PromptBlock(
                    name="tool_policy",
                    content=build_tool_policy_section(),
                    order=start_order + 30,
                    metadata={"cache_partition": "static", "cacheable": True},
                )
            )
            style_order = start_order + 40
        else:
            style_order = start_order + 30

        blocks.extend(
            [
                PromptBlock(name="tone_style", content=build_tone_style_section(), order=style_order, metadata={"cache_partition": "static", "cacheable": True}),
                PromptBlock(
                    name="output_efficiency",
                    content=build_output_efficiency_section(),
                    order=style_order + 10,
                    metadata={"cache_partition": "static", "cacheable": True},
                ),
            ]
        )
        return blocks

    def get_tool_catalog_prompt(self, agent:BasicAgent) -> str:
        if not agent.tool_registry:
            return ""
        try:
            tool_descriptions = agent.tool_registry.get_tools_description()
        except Exception as exc:
            logger.error(f"获取工具描述失败: {exc}")
            return "（工具描述获取失败）"
        return format_tool_catalog(tool_descriptions)

    def get_tool_inventory_prompt(self, agent:BasicAgent, *, include_parameters: bool = False) -> str:
        if not agent.tool_registry:
            return ""
        try:
            tool_descriptions = agent.tool_registry.list_tool_descriptors(
                stable=True,
                include_parameters=include_parameters,
            )
        except Exception as exc:
            logger.error(f"获取工具描述失败: {exc}")
            return "（工具描述获取失败）"
        return format_tool_inventory(tool_descriptions, include_parameters=include_parameters)

    def build_tool_inventory_block(self, agent:BasicAgent, order: int) -> PromptBlock | None:
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
        deferred_note = ""
        if getattr(getattr(agent, "config", None), "tool_schema_mode", "full") == "deferred":
            deferred_note = (
                "当前使用按需工具 schema 模式。先根据下面的概览判断需要哪个工具；"
                "若该工具当前尚未出现在 tools 集合中，先调用 `tool_schema_tool` 展开，再在后续回合调用目标工具。\n\n"
            )

        return PromptBlock(
            name="tool_inventory",
            content=f"{title}\n{deferred_note}{content}",
            order=order,
            metadata={"cache_partition": "session", "cacheable": True, "request_layer": "reminder"},
        )

    def build_shared_prompt_blocks(
        self,
        agent:BasicAgent,
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
                    metadata={"cache_partition": "session", "cacheable": True},
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
                    metadata={"cache_partition": "static", "cacheable": True},
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
                    metadata={"cache_partition": "session", "cacheable": True, "request_layer": "reminder"},
                )
            )
            order += 10

        runtime_reminder_blocks = agent.build_runtime_reminder_prompt_blocks(start_order=order)
        if runtime_reminder_blocks:
            blocks.extend(runtime_reminder_blocks)
            order = max(order, max(block.order for block in runtime_reminder_blocks) + 10)

        memory_prompt = agent._build_memory_prompt() if include_memory else ""
        if memory_prompt:
            blocks.append(
                PromptBlock(
                    name="memory",
                    content=memory_prompt,
                    order=order,
                    metadata={"cache_partition": "dynamic", "cacheable": False},
                )
            )
            order += 10

        mailbox_prompt = agent._build_mailbox_prompt()
        if mailbox_prompt:
            blocks.append(
                PromptBlock(
                    name="mailbox",
                    content=mailbox_prompt,
                    order=order,
                    metadata={"cache_partition": "dynamic", "cacheable": False},
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
                        metadata={"cache_partition": "session", "cacheable": True},
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
