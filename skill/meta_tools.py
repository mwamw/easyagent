"""
Skill 元工具 — LLM 动态发现与调用 Skill

优先模式：
1. SkillDiscoveryTool  — 查看可用 Skill listing
2. SkillTool           — 按需注入 Skill 正文，并在必要时挂载工具/上下文

兼容模式：
3. LoadSkillTool       — 动态加载 Skill 到当前 Agent（legacy mount）
4. UnloadSkillTool     — 卸载不需要的 Skill

推荐调用流程::

    用户: "帮我算一下 2^100"
    LLM:  先查看 system prompt 中的 skill listing
    LLM:  调用 skill_tool(skill_name="calculator")
    系统:  返回 "已注入 Skill 'calculator' 的正文指导 ..."
    LLM:  使用 calculator_tool 完成计算
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool, ToolResult

if TYPE_CHECKING:
    from skill.registry import SkillRegistry
    from skill.manager import SkillManager

logger = logging.getLogger(__name__)


class SkillDiscoveryParams(BaseModel):
    query: str = Field(default="", description="按关键词筛选可用 Skill，可为空")


class SkillRunParams(BaseModel):
    skill_name: str = Field(description="要调用的 Skill 注册名称")
    skill_arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "传给 Skill 的额外参数。当前主要用于 MCPPromptSkill 的 prompt 参数；"
            "例如 {\"language\": \"中文\"}。"
        ),
    )


class LoadSkillParams(BaseModel):
    skill_name: str = Field(
        description="要长期加载到当前 Agent 的 Skill 注册名称",
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
                "按关键词补充检索可用的额外技能包(Skill)列表。"
                "优先使用 system prompt 中已有的 skill listing；只有当 listing 不足以判断、"
                "需要按关键词筛选，或怀疑可用 Skill 集合发生变化时，再调用此工具。"
            ),
            parameters=SkillDiscoveryParams,
        )
        self._registry = registry

    def run(self, parameters: dict) -> ToolResult:
        query = str(parameters.get("query", "")).strip()
        skills = self._registry.search(query=query) if query else self._registry.list_available()
        if not skills:
            return ToolResult.success("当前没有可用的额外 Skill。")
        return ToolResult.success(
            structured_data=skills,
            metadata={"query": query},
        )


class SkillTool(Tool):
    """
    Skill 调用工具 — 按需注入 Skill 正文，并在必要时挂载能力。

    该工具会：
    1. 从 Registry 创建 Skill 实例
    2. 将其工具/上下文挂载到当前 Agent（如需要）
    3. 将 Skill 正文作为工具结果返回，供下一轮模型继续执行
    """

    def __init__(self, registry: "SkillRegistry", manager: "SkillManager", loaded_tracker: set):
        super().__init__(
            name="skill_tool",
            description=(
                "按需调用一个 Skill。优先根据 system prompt 里的 skill listing 直接调用此工具，"
                "而不是先调用 skill_discovery_tool。调用后不会返回 Skill 的详细正文；"
                "详细正文会以当前轮临时上下文的形式注入到后续推理链中；"
                "如该 Skill 提供工具，也会仅为当前轮临时挂载，并在当前轮结束后自动完全卸载。"
            ),
            parameters=SkillRunParams,
        )
        self._registry = registry
        self._manager = manager
        self._loaded_tracker = loaded_tracker

    def run(self, parameters: dict) -> ToolResult:
        skill_name = parameters.get("skill_name", "")
        skill_arguments = parameters.get("skill_arguments", {}) or {}
        if not skill_name:
            return ToolResult.error("错误：必须指定 skill_name", error_type="invalid_parameters")
        if not isinstance(skill_arguments, dict):
            return ToolResult.error(
                "错误：skill_arguments 必须是对象/字典。",
                error_type="invalid_parameters",
            )

        if not self._registry.has(skill_name):
            available = self._registry.list_available_names()
            return ToolResult.error(
                f"错误：Skill '{skill_name}' 未在注册中心中找到。"
                f"可用 Skill: {available}",
                error_type="not_found",
            )

        try:
            manifest = self._registry.get_manifest(skill_name)
            existed_before = self._manager.has_skill(skill_name)
            was_active_before = existed_before and self._manager.is_active(skill_name)
            create_kwargs: dict[str, Any] = {}
            if skill_arguments and manifest.source_type == "mcp_prompt":
                create_kwargs["prompt_arguments"] = {
                    str(key): str(value) for key, value in skill_arguments.items()
                }

            if self._manager.has_skill(skill_name):
                skill = self._manager.get_skill(skill_name)
                if skill_arguments and hasattr(skill, "set_prompt_arguments"):
                    skill.set_prompt_arguments(skill_arguments)  # type: ignore[attr-defined]
                if not self._manager.is_active(skill_name):
                    self._manager.activate(skill_name, tool_visibility="runtime")
            else:
                skill = self._registry.create(skill_name, **create_kwargs)
                self._manager.register(skill, auto_activate=False)
                if not self._manager.is_active(skill_name):
                    self._manager.activate(skill_name, tool_visibility="runtime")

            if skill.get_exposure_mode() == "on_demand":
                self._manager.mark_temporary_skill_mount(
                    skill_name,
                    created=not existed_before,
                    was_active=bool(was_active_before),
                )
            self._manager.record_invoked_skill(skill)

            body = skill.get_body_prompt().strip()
            if not body:
                body = "（该 Skill 未提供额外正文指令）"
            self._manager.set_runtime_skill_context(skill, body, source="skill_tool")

            return ToolResult.success(
                f"已注入 Skill `{skill_name}`。\n"
                "该 Skill 的详细正文已注入当前 invoke 的后续推理链，请直接基于当前新增上下文继续执行。\n",
                metadata={
                    "skill_name": skill_name,
                    "skill_arguments": dict(skill_arguments),
                    "exposure_mode": skill.get_exposure_mode(),
                    "execution_mode": skill.get_execution_mode(),
                },
            )
        except Exception as e:
            logger.error("调用 Skill '%s' 失败: %s", skill_name, e)
            return ToolResult.error(
                f"调用 Skill '{skill_name}' 失败: {e}",
                error_type="skill_invoke_failed",
                metadata={"skill_name": skill_name},
            )


class LoadSkillTool(Tool):
    """
    Skill 加载工具 — 兼容路径：动态加载 Skill 到当前 Agent

    仅在明确需要将某个 Skill 长期挂载到当前 Agent 时使用。
    对于大多数临时任务，优先使用 skill_tool。
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
                "兼容接口：将一个 Skill 长期加载到当前 Agent。"
                "只有在你明确需要让某个 Skill 在后续多轮持续保持激活时才使用；"
                "大多数场景优先使用 skill_tool。"
            ),
            parameters=LoadSkillParams,
        )
        self._registry = registry
        self._manager = manager
        self._loaded_tracker = loaded_tracker

    def run(self, parameters: dict) -> ToolResult:
        skill_name = parameters.get("skill_name", "")

        if not skill_name:
            return ToolResult.error("错误：必须指定 skill_name", error_type="invalid_parameters")

        # 检查是否已加载
        if self._manager.has_skill(skill_name):
            if self._manager.is_active(skill_name):
                return ToolResult.success(f"Skill '{skill_name}' 已经加载且处于激活状态，无需重复加载。")
            else:
                try:
                    self._manager.activate(skill_name)
                    self._loaded_tracker.add(skill_name)
                    skill = self._manager.get_skill(skill_name)
                    tool_names = skill.get_tool_names()
                    return ToolResult.success(
                        f"Skill '{skill_name}' 已重新激活。"
                        f"可用工具: {tool_names}",
                        metadata={"skill_name": skill_name, "tool_names": tool_names},
                    )
                except Exception as e:
                    return ToolResult.error(
                        f"激活 Skill '{skill_name}' 失败: {e}",
                        error_type="skill_activate_failed",
                        metadata={"skill_name": skill_name},
                    )

        # 从 Registry 创建实例
        if not self._registry.has(skill_name):
            available = self._registry.list_available_names()
            return ToolResult.error(
                f"错误：Skill '{skill_name}' 未在注册中心中找到。"
                f"可用 Skill: {available}",
                error_type="not_found",
            )

        try:
            skill = self._registry.create(skill_name)
            self._manager.register(skill, auto_activate=True, tool_visibility="resident")
            self._loaded_tracker.add(skill_name)

            tool_names = skill.get_tool_names()
            return ToolResult.success(
                f"成功加载 Skill '{skill_name}'。"
                f"新增工具: {tool_names}",
                metadata={"skill_name": skill_name, "tool_names": tool_names},
            )
        except Exception as e:
            logger.error("加载 Skill '%s' 失败: %s", skill_name, e)
            return ToolResult.error(
                f"加载 Skill '{skill_name}' 失败: {e}",
                error_type="skill_load_failed",
                metadata={"skill_name": skill_name},
            )


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
                "兼容接口：卸载你长期动态加载的技能包(Skill)。"
                "当不再需要某个通过 load_skill_tool 挂载的技能时，调用此工具卸载。"
                "注意：你只能卸载自己加载过的技能。"
            ),
            parameters=UnloadSkillParams,
        )
        self._manager = manager
        self._loaded_tracker = loaded_tracker

    def run(self, parameters: dict) -> ToolResult:
        skill_name = parameters.get("skill_name", "")

        if not skill_name:
            return ToolResult.error("错误：必须指定 skill_name", error_type="invalid_parameters")

        if skill_name not in self._loaded_tracker:
            return ToolResult.error(
                f"错误：你只能卸载自己加载过的技能！"
                f"你目前动态加载过的技能有: {list(self._loaded_tracker) if self._loaded_tracker else '无'}",
                error_type="not_allowed",
            )

        if not self._manager.has_skill(skill_name):
            active = self._manager.active_skill_names
            return ToolResult.error(
                f"错误：Skill '{skill_name}' 未加载。"
                f"当前活跃 Skill: {active}",
                error_type="not_found",
            )

        try:
            # 获取工具名以便反馈
            skill = self._manager.get_skill(skill_name)
            tool_names = skill.get_tool_names()

            self._manager.unregister(skill_name)
            self._loaded_tracker.remove(skill_name)

            return ToolResult.success(
                f"成功卸载 Skill '{skill_name}'。"
                f"已移除工具: {tool_names}",
                metadata={"skill_name": skill_name, "tool_names": tool_names},
            )
        except Exception as e:
            logger.error("卸载 Skill '%s' 失败: %s", skill_name, e)
            return ToolResult.error(
                f"卸载 Skill '{skill_name}' 失败: {e}",
                error_type="skill_unload_failed",
                metadata={"skill_name": skill_name},
            )

# ==================== MetaSkill 封装 ====================

from skill.base import BaseSkill, SkillConfig

class MetaSkill(BaseSkill):
    """
    负责动态管理技能能力的超级技能
    提供 skill_discovery_tool, skill_tool, load_skill_tool, unload_skill_tool
    """

    def __init__(self, registry: "SkillRegistry", manager: "SkillManager"):
        config = SkillConfig(
            name="meta_skill",
            description="动态技能加载和管理",
            priority=100, # 高优先级
            listing_description="查看可用 Skills，并按需调用/挂载它们",
            when_to_use="当前能力不足，需要额外技能目录或按需 Skill 指令时",
        )
        super().__init__(config)
        self._registry = registry
        self._manager = manager
        self._manager.bind_registry(registry)
        # 跟踪此 Agent 运行时动态加载的技能
        self._dynamically_loaded_skills = set()
        
        self._tools = [
            SkillDiscoveryTool(registry),
            SkillTool(registry, manager, self._dynamically_loaded_skills),
            LoadSkillTool(registry, manager, self._dynamically_loaded_skills),
            UnloadSkillTool(manager, self._dynamically_loaded_skills),
        ]

    def get_tools(self) -> List["Tool"]:
        return self._tools

    def get_prompt(self) -> str:
        return (
            "## 动态技能管理工具\n"
            "优先查看主系统提示词中的 skill listing，并直接使用 `skill_tool` 按需调用 Skill。"
            "`skill_discovery_tool` 只用于补充检索；`load_skill_tool` / `unload_skill_tool` 仅用于长期挂载兼容场景。"
        )
