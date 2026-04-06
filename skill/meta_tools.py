"""
Skill 元工具 — LLM 动态按需加载 Skill（模式 B）

提供 3 个元工具，让 LLM 在运行时自主发现、加载、卸载 Skill：

1. SkillDiscoveryTool  — 按关键词搜索可用 Skill
2. LoadSkillTool       — 动态加载 Skill 到当前 Agent
3. UnloadSkillTool     — 卸载不需要的 Skill

典型调用流程::

    用户: "帮我算一下 2^100"
    LLM:  调用 skill_discovery_tool(query="math calculation")
    系统:  返回 [{"name": "calculator", ...}]
    LLM:  调用 load_skill_tool(skill_name="calculator")
    系统:  返回 "成功加载 Skill 'calculator'，新增工具: ['calculator_tool']"
    LLM:  使用 calculator_tool 完成计算
    LLM:  调用 unload_skill_tool(skill_name="calculator")
"""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool

if TYPE_CHECKING:
    from skill.registry import SkillRegistry
    from skill.manager import SkillManager

logger = logging.getLogger(__name__)


class SkillDiscoveryParams(BaseModel):
    pass



class LoadSkillParams(BaseModel):
    skill_name: str = Field(
        description="要加载的 Skill 注册名称（从 skill_discovery_tool 的返回结果中获取）",
    )


class UnloadSkillParams(BaseModel):
    skill_name: str = Field(
        description="要卸载的 Skill 名称",
    )


# ==================== 元工具实现 ====================


class SkillDiscoveryTool(Tool):
    """
    Skill 发现工具 — 按关键词搜索可用 Skill

    LLM 在不知道该用什么能力时，调用此工具搜索匹配的 Skill。
    返回可加载的 Skill 列表及其描述。
    """

    def __init__(self, registry: "SkillRegistry"):
        """
        Args:
            registry: SkillRegistry 全局注册中心实例
        """
        super().__init__(
            name="skill_discovery_tool",
            description=(
                "获取所有可用的额外技能包(Skill)列表及描述。"
                "当你发现当前工具箱中没有合适的工具时，调用此工具获取可加载的技能列表。"
            ),
            parameters=SkillDiscoveryParams,
        )
        self._registry = registry

    def run(self, parameters: dict) -> str:
        skills_desc = self._registry.get_skills_description()
        if not skills_desc:
            return "当前没有可用的额外 Skill。"

        lines = ["当前可用的额外技能如下："]
        for name, desc in skills_desc.items():
            lines.append(f"- 【{name}】: {desc}")

        return "\n".join(lines)


class LoadSkillTool(Tool):
    """
    Skill 加载工具 — 动态加载 Skill 到当前 Agent

    LLM 从 skill_discovery_tool 的返回结果中选择一个 Skill，
    调用此工具将其加载到当前 Agent 的工具箱中。
    """

    def __init__(self, registry: "SkillRegistry", manager: "SkillManager", loaded_tracker: set):
        """
        Args:
            registry: SkillRegistry 全局注册中心实例
            manager: 当前 Agent 的 SkillManager 实例
            loaded_tracker: 用于跟踪动态加载的 Skill 的集合
        """
        super().__init__(
            name="load_skill_tool",
            description=(
                "加载一个技能包(Skill)到当前工具箱。"
                "请先通过 skill_discovery_tool 获取可用 Skill，"
                "然后使用返回的 name 调用此工具加载。"
            ),
            parameters=LoadSkillParams,
        )
        self._registry = registry
        self._manager = manager
        self._loaded_tracker = loaded_tracker

    def run(self, parameters: dict) -> str:
        skill_name = parameters.get("skill_name", "")

        if not skill_name:
            return "错误：必须指定 skill_name"

        # 检查是否已加载
        if self._manager.has_skill(skill_name):
            if self._manager.is_active(skill_name):
                return f"Skill '{skill_name}' 已经加载且处于激活状态，无需重复加载。"
            else:
                try:
                    self._manager.activate(skill_name)
                    self._loaded_tracker.add(skill_name)
                    skill = self._manager.get_skill(skill_name)
                    tool_names = skill.get_tool_names()
                    return (
                        f"Skill '{skill_name}' 已重新激活。"
                        f"可用工具: {tool_names}"
                    )
                except Exception as e:
                    return f"激活 Skill '{skill_name}' 失败: {e}"

        # 从 Registry 创建实例
        if not self._registry.has(skill_name):
            available = self._registry.list_available_names()
            return (
                f"错误：Skill '{skill_name}' 未在注册中心中找到。"
                f"可用 Skill: {available}"
            )

        try:
            skill = self._registry.create(skill_name)
            self._manager.register(skill)  # auto_activate 默认为 True
            self._loaded_tracker.add(skill_name)

            tool_names = skill.get_tool_names()
            return (
                f"成功加载 Skill '{skill_name}'。"
                f"新增工具: {tool_names}"
            )
        except Exception as e:
            logger.error("加载 Skill '%s' 失败: %s", skill_name, e)
            return f"加载 Skill '{skill_name}' 失败: {e}"


class UnloadSkillTool(Tool):
    """
    Skill 卸载工具 — 卸载不需要的 Skill

    LLM 在完成任务后，可以调用此工具卸载不再需要的 Skill，
    释放上下文空间。
    """

    def __init__(self, manager: "SkillManager", loaded_tracker: set):
        """
        Args:
            manager: 当前 Agent 的 SkillManager 实例
            loaded_tracker: 用于跟踪动态加载的 Skill 的集合
        """
        super().__init__(
            name="unload_skill_tool",
            description=(
                "卸载你动态加载的技能包(Skill)。"
                "当不再需要某个技能时，调用此工具卸载。注意：你只能卸载自己加载过的技能。"
            ),
            parameters=UnloadSkillParams,
        )
        self._manager = manager
        self._loaded_tracker = loaded_tracker

    def run(self, parameters: dict) -> str:
        skill_name = parameters.get("skill_name", "")

        if not skill_name:
            return "错误：必须指定 skill_name"

        if skill_name not in self._loaded_tracker:
            return (
                f"错误：你只能卸载自己加载过的技能！"
                f"你目前动态加载过的技能有: {list(self._loaded_tracker) if self._loaded_tracker else '无'}"
            )

        if not self._manager.has_skill(skill_name):
            active = self._manager.active_skill_names
            return (
                f"错误：Skill '{skill_name}' 未加载。"
                f"当前活跃 Skill: {active}"
            )

        try:
            # 获取工具名以便反馈
            skill = self._manager.get_skill(skill_name)
            tool_names = skill.get_tool_names()

            self._manager.unregister(skill_name)
            self._loaded_tracker.remove(skill_name)

            return (
                f"成功卸载 Skill '{skill_name}'。"
                f"已移除工具: {tool_names}"
            )
        except Exception as e:
            logger.error("卸载 Skill '%s' 失败: %s", skill_name, e)
            return f"卸载 Skill '{skill_name}' 失败: {e}"

# ==================== MetaSkill 封装 ====================

from skill.base import BaseSkill, SkillConfig

class MetaSkill(BaseSkill):
    """
    负责动态管理技能能力的超级技能
    提供 skill_discovery_tool, load_skill_tool, unload_skill_tool
    """

    def __init__(self, registry: "SkillRegistry", manager: "SkillManager"):
        config = SkillConfig(
            name="meta_skill",
            description="动态技能加载和管理",
            priority=100, # 高优先级
        )
        super().__init__(config)
        self._registry = registry
        self._manager = manager
        # 跟踪此 Agent 运行时动态加载的技能
        self._dynamically_loaded_skills = set()
        
        self._tools = [
            SkillDiscoveryTool(registry),
            LoadSkillTool(registry, manager, self._dynamically_loaded_skills),
            UnloadSkillTool(manager, self._dynamically_loaded_skills),
        ]

    def get_tools(self) -> List["Tool"]:
        return self._tools

    def get_prompt(self) -> str:
        return (
            "## 动态技能组装能力\n"
            "如果你发现完成当前任务需要某些你尚未拥有的额外技能或工具，请使用 `skill_discovery_tool` 查看系统中可用的额外技能包列表。\n"
            "当你确定想使用某个技能时，通过 `load_skill_tool` 将其装载，装载后你将获得对应的新能力。\n"
            "完成任务之后，为了释放上下文不再占据资源，请使用 `unload_skill_tool` 卸载你刚才加载过的技能。"
        )


