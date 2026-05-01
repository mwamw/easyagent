"""
SkillManager — Skill 管理器

管理 Skill 的注册、激活、停用，以及动态注入 Tool / Prompt / ContextSource 到 Agent。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from core.cache_policy import CacheableBlock
from .base import BaseSkill, SkillManifest
from prompt import (
    build_skill_listing_section,
    build_skill_policy_section,
    build_runtime_skill_context_section,
    build_skills_prompt_section,
)

if TYPE_CHECKING:
    from core.agent import BaseAgent
    from skill.registry import SkillRegistry
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
        # 记录本轮/近期按需调用过的 skill，供后续注入和调试
        self._invoked_skill_manifests: Dict[str, SkillManifest] = {}
        # 记录按需 skill 的临时正文上下文，不写入长期 system prompt
        self._runtime_skill_contexts: Dict[str, dict[str, Any]] = {}
        # 记录本轮通过 skill_tool 临时挂载的 skill，invoke 结束后恢复原状态
        self._temporary_skill_mounts: Dict[str, dict[str, bool]] = {}
        # 可选绑定 registry，用于在主 prompt 中展示完整 skill listing
        self._registry: Optional["SkillRegistry"] = None

    def bind_agent(self, agent: "BaseAgent") -> None:
        """
        绑定到 Agent 实例

        Args:
            agent: BaseAgent 实例
        """
        self._agent = agent
        logger.debug("SkillManager 已绑定到 Agent '%s'", getattr(agent, "name", "unknown"))

    def bind_registry(self, registry: "SkillRegistry") -> None:
        """绑定 SkillRegistry，用于生成完整 skill listing。"""
        self._registry = registry

    # ==================== Skill 注册 / 注销 ====================

    def register(
        self,
        skill: BaseSkill,
        *,
        auto_activate: Optional[bool] = None,
        tool_visibility: str = "resident",
    ) -> "SkillManager":
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
        should_auto_activate = skill.config.auto_activate if auto_activate is None else auto_activate
        if should_auto_activate:
            self.activate(skill.name, tool_visibility=tool_visibility)

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
        self._invoked_skill_manifests.pop(name, None)
        logger.info("📦 注销 Skill '%s'", name)

    # ==================== 激活 / 停用 ====================

    def activate(self, name: str, *, tool_visibility: str = "resident") -> None:
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
        tool_names = self._inject_tools(skill, tool_visibility=tool_visibility)
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
        skills.sort(key=lambda s: (-s.priority, s.name))
        return skills

    def get_active_resident_skills(self) -> List[BaseSkill]:
        """获取所有激活且以 resident 方式暴露的 Skill。"""
        return [
            skill for skill in self.get_active_skills()
            if skill.get_exposure_mode() == "resident"
        ]

    def get_active_on_demand_skills(self) -> List[BaseSkill]:
        """获取所有激活但不进入 system prompt 的 Skill。"""
        return [
            skill for skill in self.get_active_skills()
            if skill.get_exposure_mode() == "on_demand"
        ]

    def get_all_skills(self) -> List[BaseSkill]:
        """获取所有已注册的 Skill"""
        return list(self._skills.values())

    def _collect_skill_manifests(self) -> List[SkillManifest]:
        """收集当前 Agent 与外部 registry 可见的 Skill manifests。"""
        manifests: dict[str, SkillManifest] = {}

        for skill in self._skills.values():
            manifests[skill.name] = skill.build_manifest()

        if self._registry is not None:
            try:
                for manifest in self._registry.list_manifests():
                    manifests.setdefault(manifest.name, manifest.model_copy(deep=True))
            except Exception as e:
                logger.warning("从 SkillRegistry 收集 manifest 失败: %s", e)

        result = list(manifests.values())
        result.sort(key=lambda item: (-item.priority, item.name))
        return result

    def get_on_demand_skill_manifests(self) -> List[SkillManifest]:
        """获取所有当前可见的 on-demand Skill manifests。"""
        return [
            manifest for manifest in self._collect_skill_manifests()
            if manifest.exposure_mode == "on_demand" and manifest.name != "meta_skill"
        ]

    def get_skill_cache_lifecycle(self, name: str) -> str:
        skill = self.get_skill(name)
        return skill.get_cache_lifecycle()

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

    def build_resident_skills_prompt(
        self,
        exclude_names: Optional[Iterable[str]] = None,
    ) -> str:
        """
        将所有激活且 exposure_mode=resident 的 Skill 正文拼接为一个完整的技能提示。

        按 priority 降序排列，高优先级 Skill 的 prompt 在前。

        Returns:
            拼接后的 prompt 文本，无激活 Skill 时返回空字符串
        """
        excluded = set(exclude_names or [])
        active_skills = self.get_active_resident_skills()
        if not active_skills:
            return ""

        prompt_parts = []
        for skill in active_skills:
            if skill.name in excluded:
                continue
            if skill.get_cache_lifecycle() == "turn" and skill.get_exposure_mode() != "resident":
                continue
            try:
                skill_prompt = skill.get_body_prompt()
                if skill_prompt and skill_prompt.strip():
                    prompt_parts.append(skill_prompt.strip())
            except Exception as e:
                logger.warning("Skill '%s' 获取 prompt 失败: %s", skill.name, e)

        if not prompt_parts:
            return ""

        return build_skills_prompt_section(prompt_parts)

    def build_skills_prompt(
        self,
        exclude_names: Optional[Iterable[str]] = None,
    ) -> str:
        """向后兼容：仅聚合 resident Skills 的正文。"""
        return self.build_resident_skills_prompt(exclude_names=exclude_names)

    def build_skill_policy_prompt(self) -> str:
        """构建按需 Skill 使用规则。"""
        if not self.get_on_demand_skill_manifests():
            return ""
        return build_skill_policy_section()

    def build_skill_listing_prompt(self) -> str:
        """构建按需 Skill listing。"""
        manifests = self.get_on_demand_skill_manifests()
        if not manifests:
            return ""
        return build_skill_listing_section(
            [manifest.model_dump(mode="json") for manifest in manifests]
        )

    def record_invoked_skill(self, skill: BaseSkill) -> None:
        """记录按需调用过的 Skill manifest。"""
        self._invoked_skill_manifests[skill.name] = skill.build_manifest()

    def set_runtime_skill_context(
        self,
        skill: BaseSkill,
        body: str,
        *,
        source: str = "skill_tool",
    ) -> None:
        """记录临时 Skill 正文上下文。"""
        self._runtime_skill_contexts[skill.name] = {
            "body": body.strip(),
            "source": source,
            "manifest": skill.build_manifest(),
        }

    def get_invoked_skill_manifests(self) -> List[SkillManifest]:
        """返回按需调用过的 Skill manifest。"""
        manifests = list(self._invoked_skill_manifests.values())
        manifests.sort(key=lambda item: (-item.priority, item.name))
        return manifests

    def clear_invoked_skill(self, name: str) -> None:
        """清理指定按需调用记录。"""
        self._invoked_skill_manifests.pop(name, None)

    def build_runtime_skill_context_prompt(self) -> str:
        """构建当前回合的临时 Skill 正文上下文。"""
        if not self._runtime_skill_contexts:
            return ""
        items = list(self._runtime_skill_contexts.items())
        items.sort(key=lambda item: (-item[1]["manifest"].priority, item[0]))
        payloads: list[dict[str, Any]] = []
        for skill_name, payload in items:
            manifest: SkillManifest = payload["manifest"]
            payloads.append(
                {
                    "name": skill_name,
                    "source": payload.get("source", "skill_tool"),
                    "when_to_use": manifest.when_to_use,
                    "source_path": manifest.source_path,
                    "tool_names": manifest.tool_names,
                    "body": payload.get("body", ""),
                }
            )
        return build_runtime_skill_context_section(payloads)

    def build_runtime_skill_context_blocks(self) -> list[CacheableBlock]:
        if not self._runtime_skill_contexts:
            return []
        items = list(self._runtime_skill_contexts.items())
        items.sort(key=lambda item: (-item[1]["manifest"].priority, item[0]))
        blocks: list[CacheableBlock] = []
        for skill_name, payload in items:
            manifest: SkillManifest = payload["manifest"]
            block_name = f"skill:{skill_name}"
            block_content = build_runtime_skill_context_section(
                [
                    {
                        "name": skill_name,
                        "source": payload.get("source", "skill_tool"),
                        "when_to_use": manifest.when_to_use,
                        "source_path": manifest.source_path,
                        "tool_names": manifest.tool_names,
                        "body": payload.get("body", ""),
                    }
                ]
            )
            blocks.append(
                CacheableBlock(
                    name=block_name,
                    content=block_content,
                    partition="session",
                    cacheable=True,
                    reason="runtime_skill_context",
                    metadata={
                        "request_layer": "on_demand_expansion",
                        "skill_name": skill_name,
                        "skill_source": payload.get("source", "skill_tool"),
                    },
                )
            )
        return blocks

    def has_runtime_skill_context(self) -> bool:
        """当前是否存在临时 Skill 正文上下文。"""
        return bool(self._runtime_skill_contexts)

    def clear_runtime_skill_context(self, name: Optional[str] = None) -> None:
        """清理临时 Skill 正文上下文。"""
        if name is None:
            self._runtime_skill_contexts.clear()
            return
        self._runtime_skill_contexts.pop(name, None)

    def mark_temporary_skill_mount(
        self,
        name: str,
        *,
        created: bool,
        was_active: bool,
    ) -> None:
        """记录本轮临时挂载的 Skill，以便 invoke 结束后恢复。"""
        payload = self._temporary_skill_mounts.get(name)
        if payload is None:
            self._temporary_skill_mounts[name] = {
                "created": created,
                "was_active": was_active,
            }
            return

        payload["created"] = payload["created"] or created
        payload["was_active"] = payload["was_active"] or was_active

    def cleanup_temporary_skill_mounts(self) -> None:
        """回收本轮通过 skill_tool 临时挂载的 Skills。"""
        if not self._temporary_skill_mounts:
            return

        pending = list(self._temporary_skill_mounts.items())
        self._temporary_skill_mounts.clear()
        for name, payload in reversed(pending):
            created = bool(payload.get("created"))
            was_active = bool(payload.get("was_active"))
            try:
                if created:
                    if self.has_skill(name):
                        self.unregister(name)
                    continue

                if not was_active and self.is_active(name):
                    self.deactivate(name)
            except Exception as e:
                logger.warning("清理临时 Skill '%s' 失败: %s", name, e)

    def clear_ephemeral_state(self) -> None:
        """清理当前轮的临时 Skill 状态。"""
        self.clear_runtime_skill_context()
        self.cleanup_temporary_skill_mounts()

    def expose_runtime_skill_tools(self, skill_name: str) -> list[str]:
        if self._agent is None:
            return []
        registry = getattr(self._agent, "tool_registry", None)
        if registry is None:
            return []
        tool_names = list(self._skill_tool_names.get(skill_name, []))
        if not tool_names:
            return []
        if getattr(getattr(self._agent, "config", None), "tool_schema_mode", "full") == "deferred":
            registry.expand_deferred_tools(tool_names)
        return tool_names

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

    def _inject_tools(self, skill: BaseSkill, *, tool_visibility: str = "resident") -> List[str]:
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
                try:
                    tool.spec.metadata.setdefault("skill_name", skill.name)
                    tool.spec.metadata.setdefault("skill_exposure_mode", skill.get_exposure_mode())
                    tool.spec.metadata.setdefault("skill_tool_visibility", tool_visibility)
                except Exception:
                    pass
                if skill.get_exposure_mode() == "on_demand" and tool_visibility in {"runtime", "turn"}:
                    tool.mark_as_demand_skill_tool(skill.name)
                else:
                    tool.clear_demand_skill_tool()
                if not registry.has_tool(tool.name):
                    if tool_visibility == "runtime":
                        registry.mount_runtime_tool(tool)
                    elif tool_visibility == "turn":
                        registry.mount_turn_tool(tool)
                    else:
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
