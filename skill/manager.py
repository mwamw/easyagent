"""
SkillManager — Skill 管理器

管理 Skill 的注册、激活、停用，以及动态注入 Tool / Prompt / ContextSource 到 Agent。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseSkill

if TYPE_CHECKING:
    from core.agent import BaseAgent
    from Tool.ToolRegistry import ToolRegistry

logger = logging.getLogger(__name__)


class SkillManager:
    """
    Skill 管理器 — Agent 的能力管理中心

    职责:
    1. 管理 Skill 的注册 / 注销
    2. 管理 Skill 的激活 / 停用
    3. 激活时：将 Skill 的 Tool 注册到 Agent 的 ToolRegistry
    4. 停用时：将 Skill 的 Tool 从 Agent 的 ToolRegistry 移除
    5. 聚合所有激活 Skill 的 prompt 供 Agent 使用
    6. 代理 Skill 生命周期钩子

    Example::

        manager = SkillManager()
        manager.bind_agent(agent)
        manager.register(WebSearchSkill())
        manager.activate("web_search")
        prompt = manager.build_skills_prompt()
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}       # 所有已注册 Skill
        self._active_skills: Dict[str, BaseSkill] = {}  # 当前激活的 Skill
        self._agent: Optional["BaseAgent"] = None
        # 记录每个 Skill 注入的工具名，用于 deactivate 时精确移除
        self._skill_tool_names: Dict[str, List[str]] = {}
        # 记录每个 Skill 注入的 context source 名称
        self._skill_source_names: Dict[str, List[str]] = {}

    def bind_agent(self, agent: "BaseAgent") -> None:
        """
        绑定到 Agent 实例

        Args:
            agent: BaseAgent 实例
        """
        self._agent = agent
        logger.debug("SkillManager 已绑定到 Agent '%s'", getattr(agent, "name", "unknown"))

    # ==================== Skill 注册 / 注销 ====================

    def register(self, skill: BaseSkill) -> "SkillManager":
        """
        注册一个 Skill

        如果 Skill 配置了 auto_activate=True 且已绑定 Agent，会自动激活。

        Args:
            skill: BaseSkill 实例

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: Skill 名称冲突
            TypeError: 参数类型不正确
        """
        if not isinstance(skill, BaseSkill):
            raise TypeError(f"skill 必须是 BaseSkill 子类实例，收到: {type(skill).__name__}")

        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' 已存在，请先注销再重新注册")

        # 检查依赖
        self._check_dependencies(skill)

        self._skills[skill.name] = skill
        logger.info("📦 注册 Skill '%s' (v%s)", skill.name, skill.config.version)

        # 自动激活
        if skill.config.auto_activate:
            self.activate(skill.name)

        return self

    def unregister(self, name: str) -> None:
        """
        注销一个 Skill（先停用再移除）

        Args:
            name: Skill 名称

        Raises:
            KeyError: Skill 不存在
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' 不存在")

        # 如果是激活状态，先停用
        if name in self._active_skills:
            self.deactivate(name)

        del self._skills[name]
        logger.info("📦 注销 Skill '%s'", name)

    # ==================== 激活 / 停用 ====================

    def activate(self, name: str) -> None:
        """
        激活指定 Skill

        激活操作:
        1. 将 Skill 的 Tools 注册到 Agent 的 ToolRegistry
        2. 将 Skill 的 ContextSource 注册到 Agent 的 ContextManager
        3. 调用 Skill 的 on_activate 钩子
        4. 标记为激活状态

        Args:
            name: Skill 名称

        Raises:
            KeyError: Skill 不存在
            RuntimeError: Skill 已处于激活状态
        """
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' 不存在，请先注册")

        if name in self._active_skills:
            logger.warning("Skill '%s' 已处于激活状态", name)
            return

        skill = self._skills[name]

        # 检查依赖的 Skill 是否已激活
        for dep in skill.config.dependencies:
            if dep not in self._active_skills:
                logger.warning(
                    "Skill '%s' 依赖 '%s'，但 '%s' 尚未激活。尝试自动激活...",
                    name, dep, dep,
                )
                if dep in self._skills:
                    self.activate(dep)
                else:
                    raise RuntimeError(
                        f"Skill '{name}' 依赖 '{dep}'，但 '{dep}' 未注册"
                    )

        # 1. 注册 Tools
        tool_names = self._inject_tools(skill)
        self._skill_tool_names[name] = tool_names

        # 2. 注册 ContextSources
        source_names = self._inject_context_sources(skill)
        self._skill_source_names[name] = source_names

        # 3. 生命周期钩子
        try:
            skill.on_activate(self._agent)
        except Exception as e:
            logger.error("Skill '%s' on_activate 失败: %s", name, e)

        # 4. 标记激活
        skill._is_active = True
        self._active_skills[name] = skill

        logger.info(
            "✅ 激活 Skill '%s' (工具: %s)",
            name, tool_names,
        )

    def deactivate(self, name: str) -> None:
        """
        停用指定 Skill

        停用操作:
        1. 从 Agent 的 ToolRegistry 移除 Skill 的 Tools
        2. 调用 Skill 的 on_deactivate 钩子
        3. 标记为停用状态

        Args:
            name: Skill 名称

        Raises:
            KeyError: Skill 不存在或未激活
        """
        if name not in self._active_skills:
            raise KeyError(f"Skill '{name}' 未激活或不存在")

        skill = self._active_skills[name]

        # 检查是否有其他激活的 Skill 依赖此 Skill
        for other_name, other_skill in self._active_skills.items():
            if other_name != name and name in other_skill.config.dependencies:
                logger.warning(
                    "Skill '%s' 被 '%s' 依赖，先停用 '%s'",
                    name, other_name, other_name,
                )
                self.deactivate(other_name)

        # 1. 移除 Tools
        self._remove_tools(name)

        # 2. 生命周期钩子
        try:
            skill.on_deactivate(self._agent)
        except Exception as e:
            logger.error("Skill '%s' on_deactivate 失败: %s", name, e)

        # 3. 标记停用
        skill._is_active = False
        del self._active_skills[name]

        # 清理记录
        self._skill_tool_names.pop(name, None)
        self._skill_source_names.pop(name, None)

        logger.info("⏸️  停用 Skill '%s'", name)

    # ==================== 查询 ====================

    def get_skill(self, name: str) -> BaseSkill:
        """获取指定 Skill"""
        if name not in self._skills:
            available = list(self._skills.keys())
            raise KeyError(f"Skill '{name}' 不存在。已注册: {available}")
        return self._skills[name]

    def get_active_skills(self) -> List[BaseSkill]:
        """获取所有激活的 Skill（按 priority 降序排列）"""
        skills = list(self._active_skills.values())
        skills.sort(key=lambda s: s.priority, reverse=True)
        return skills

    def get_all_skills(self) -> List[BaseSkill]:
        """获取所有已注册的 Skill"""
        return list(self._skills.values())

    def has_skill(self, name: str) -> bool:
        """检查 Skill 是否已注册"""
        return name in self._skills

    def is_active(self, name: str) -> bool:
        """检查 Skill 是否已激活"""
        return name in self._active_skills

    def list_skills(self) -> List[Dict[str, Any]]:
        """返回所有 Skill 的简明信息列表"""
        return [skill.to_dict() for skill in self._skills.values()]

    @property
    def active_skill_names(self) -> List[str]:
        """返回所有激活的 Skill 名称"""
        return list(self._active_skills.keys())

    @property
    def skill_count(self) -> int:
        """已注册 Skill 数量"""
        return len(self._skills)

    @property
    def active_count(self) -> int:
        """已激活 Skill 数量"""
        return len(self._active_skills)

    # ==================== Prompt 聚合 ====================

    def build_skills_prompt(self) -> str:
        """
        将所有激活 Skill 的 prompt 片段拼接为一个完整的技能提示。

        按 priority 降序排列，高优先级 Skill 的 prompt 在前。

        Returns:
            拼接后的 prompt 文本，无激活 Skill 时返回空字符串
        """
        active_skills = self.get_active_skills()
        if not active_skills:
            return ""

        prompt_parts = []
        for skill in active_skills:
            try:
                skill_prompt = skill.get_prompt()
                if skill_prompt and skill_prompt.strip():
                    prompt_parts.append(skill_prompt.strip())
            except Exception as e:
                logger.warning("Skill '%s' 获取 prompt 失败: %s", skill.name, e)

        if not prompt_parts:
            return ""

        return "\n\n" + "\n\n".join(prompt_parts)

    # ==================== 生命周期代理 ====================

    def on_before_invoke(self, query: str) -> str:
        """
        代理调用所有激活 Skill 的 on_before_invoke

        按 priority 降序调用。每个 Skill 可以修改 query，
        修改后的 query 传递给下一个 Skill。

        Args:
            query: 原始用户输入

        Returns:
            经过所有 Skill 处理的 query
        """
        for skill in self.get_active_skills():
            try:
                query = skill.on_before_invoke(query)
            except Exception as e:
                logger.warning(
                    "Skill '%s' on_before_invoke 失败: %s", skill.name, e
                )
        return query

    def on_after_invoke(self, query: str, response: str) -> str:
        """
        代理调用所有激活 Skill 的 on_after_invoke

        按 priority 降序调用。

        Args:
            query: 原始用户输入
            response: Agent 输出

        Returns:
            经过所有 Skill 处理的 response
        """
        for skill in self.get_active_skills():
            try:
                response = skill.on_after_invoke(query, response)
            except Exception as e:
                logger.warning(
                    "Skill '%s' on_after_invoke 失败: %s", skill.name, e
                )
        return response

    # ==================== 内部方法 ====================

    def _check_dependencies(self, skill: BaseSkill) -> None:
        """检查 Skill 依赖是否满足（仅警告，不阻止注册）"""
        for dep in skill.config.dependencies:
            if dep not in self._skills:
                logger.warning(
                    "Skill '%s' 声明依赖 '%s'，但 '%s' 尚未注册",
                    skill.name, dep, dep,
                )

    def _inject_tools(self, skill: BaseSkill) -> List[str]:
        """将 Skill 的 Tools 注册到 Agent 的 ToolRegistry"""
        if self._agent is None:
            return []

        # 确保 Agent 有 ToolRegistry
        registry = getattr(self._agent, "tool_registry", None)
        if registry is None:
            from Tool.ToolRegistry import ToolRegistry
            registry = ToolRegistry()
            self._agent.tool_registry = registry  # type: ignore
            self._agent.enable_tool = True  # type: ignore

        tool_names = []
        try:
            tools = skill.get_tools()
            for tool in tools:
                if not registry.has_tool(tool.name):
                    registry.register_tool(tool)
                    tool_names.append(tool.name)
                else:
                    logger.warning(
                        "工具 '%s'（来自 Skill '%s'）名称冲突，跳过注册",
                        tool.name, skill.name,
                    )
        except Exception as e:
            logger.error("Skill '%s' get_tools() 失败: %s", skill.name, e)

        return tool_names

    def _inject_context_sources(self, skill: BaseSkill) -> List[str]:
        """将 Skill 的 ContextSources 注册到 Agent 的 ContextManager"""
        if self._agent is None:
            return []

        context_manager = getattr(self._agent, "context_manager", None)
        if context_manager is None:
            return []

        source_names = []
        try:
            sources = skill.get_context_sources()
            for source in sources:
                context_manager.add_source(source)
                source_names.append(source.source_name)
        except Exception as e:
            logger.error(
                "Skill '%s' get_context_sources() 失败: %s", skill.name, e
            )

        return source_names

    def _remove_tools(self, skill_name: str) -> None:
        """从 Agent 的 ToolRegistry 移除 Skill 的 Tools"""
        if self._agent is None:
            return

        registry = getattr(self._agent, "tool_registry", None)
        if registry is None:
            return

        tool_names = self._skill_tool_names.get(skill_name, [])
        for tool_name in tool_names:
            try:
                registry.unregister_tool(tool_name)
            except Exception as e:
                logger.warning("移除工具 '%s' 失败: %s", tool_name, e)

    def __repr__(self) -> str:
        return (
            f"SkillManager("
            f"registered={list(self._skills.keys())}, "
            f"active={list(self._active_skills.keys())})"
        )
