"""
Skill 基类模块

定义 Skill 的抽象接口、运行模式和元数据结构。
Skill 是一个「能力包」，将 Tools + Prompt + ContextSource + 配置封装为独立单元。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from Tool.BaseTool import Tool
    from context.source.base import BaseContextSource


SkillExposureMode = Literal["resident", "on_demand"]
SkillExecutionMode = Literal["mount", "inline"]
SkillCacheLifecycle = Literal["resident", "session", "turn"]


class SkillConfig(BaseModel):
    """Skill 配置数据类"""

    name: str = Field(description="Skill 唯一标识名称")
    description: str = Field(default="", description="Skill 功能描述")
    version: str = Field(default="1.0.0", description="Skill 版本号")
    tags: List[str] = Field(default_factory=list, description="标签（用于搜索/分类）")
    priority: int = Field(default=0, description="优先级，数值越大越靠前注入 prompt")
    auto_activate: bool = Field(default=True, description="注册到 SkillManager 时是否自动激活")
    dependencies: List[str] = Field(default_factory=list, description="依赖的其他 Skill 名称列表")
    listing_description: str = Field(default="", description="用于 skill listing 的简短描述")
    when_to_use: str = Field(default="", description="告诉模型何时应该调用此 Skill")
    exposure_mode: SkillExposureMode = Field(
        default="resident",
        description="resident 表示正文可常驻 system prompt，on_demand 表示正文按需注入",
    )
    execution_mode: SkillExecutionMode = Field(
        default="mount",
        description="mount 表示激活后挂载工具/上下文，inline 表示正文按需注入",
    )
    source_type: str = Field(default="python", description="Skill 来源类型，例如 python/yaml/markdown/folder")
    source_path: str = Field(default="", description="Skill 定义来源路径")
    cache_lifecycle: SkillCacheLifecycle = Field(
        default="session",
        description="控制 Skill 正文默认进入哪个 cache 分区：resident/session/turn",
    )
    extra: Dict[str, Any] = Field(default_factory=dict, description="Skill 自定义扩展配置")


class SkillManifest(BaseModel):
    """Skill 暴露给 Agent/Registry 的统一元数据。"""

    name: str
    description: str = ""
    listing_description: str = ""
    when_to_use: str = ""
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    priority: int = 0
    exposure_mode: SkillExposureMode = "resident"
    execution_mode: SkillExecutionMode = "mount"
    source_type: str = "python"
    source_path: str = ""
    cache_lifecycle: SkillCacheLifecycle = "session"
    tool_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseSkill(ABC):
    """
    Skill 抽象基类

    所有 Skill 实现应继承此类。Skill 封装了一组相关的能力，
    包括工具、prompt 指导、上下文来源和生命周期钩子。

    典型子类只需实现 get_tools() 和 get_prompt() 两个核心方法。
    `get_prompt()` 保持向后兼容，表示 Skill 正文；listing 描述和使用时机
    可以通过 config 字段或辅助方法提供。

    Example::

        class MySkill(BaseSkill):
            def __init__(self):
                config = SkillConfig(name="my_skill", description="...")
                super().__init__(config)

            def get_tools(self) -> list[Tool]:
                return [MyTool()]

            def get_prompt(self) -> str:
                return "## My Skill\\n使用 my_tool 来..."
    """

    def __init__(self, config: SkillConfig):
        if not isinstance(config, SkillConfig):
            raise TypeError(f"config 必须是 SkillConfig 类型，收到: {type(config).__name__}")
        self.config = config
        self._is_active: bool = False

    # ==================== 核心接口 ====================

    @abstractmethod
    def get_tools(self) -> List["Tool"]:
        """
        返回此 Skill 提供的所有 Tool 实例。

        SkillManager 在激活此 Skill 时，会将这些 Tool 注册到 Agent 的 ToolRegistry 中。

        Returns:
            Tool 实例列表
        """
        ...

    @abstractmethod
    def get_prompt(self) -> str:
        """
        返回此 Skill 的正文 prompt 片段。

        resident Skill 会将该片段拼接到 Agent 的 system prompt 中。
        on_demand Skill 则会在被调用时按需注入。

        Returns:
            prompt 文本片段
        """
        ...

    def get_body_prompt(self) -> str:
        """返回 Skill 正文。默认与 get_prompt() 保持一致。"""
        return self.get_prompt()

    def get_listing_description(self) -> str:
        """返回用于 skill listing 的简短描述。"""
        return (
            self.config.listing_description
            or self.config.description
            or self.name
        )

    def get_when_to_use(self) -> str:
        """返回此 Skill 的推荐使用时机。"""
        return self.config.when_to_use

    def get_exposure_mode(self) -> SkillExposureMode:
        """返回此 Skill 的暴露方式。"""
        return self.config.exposure_mode

    def get_execution_mode(self) -> SkillExecutionMode:
        """返回此 Skill 的执行方式。"""
        return self.config.execution_mode

    def get_cache_lifecycle(self) -> SkillCacheLifecycle:
        """返回此 Skill 正文的 cache 生命周期。"""
        return self.config.cache_lifecycle

    def get_context_sources(self) -> List["BaseContextSource"]:
        """
        返回此 Skill 提供的 ContextSource 列表（可选）。

        如果 Skill 需要向 ContextManager 注入额外的上下文来源，
        重写此方法即可。

        Returns:
            BaseContextSource 列表，默认返回空列表
        """
        return []

    # ==================== 生命周期钩子 ====================

    def on_activate(self, agent: Any) -> None:
        """
        Skill 被激活时调用。

        可用于初始化资源、建立连接等。

        Args:
            agent: 绑定的 BaseAgent 实例
        """
        pass

    def on_deactivate(self, agent: Any) -> None:
        """
        Skill 被停用时调用。

        可用于释放资源、关闭连接等。

        Args:
            agent: 绑定的 BaseAgent 实例
        """
        pass

    def on_before_invoke(self, query: str) -> str:
        """
        Agent invoke 前调用，可修改 query。

        用于预处理用户输入，例如自动翻译、关键词提取等。

        Args:
            query: 原始用户输入

        Returns:
            （可能修改过的）用户输入
        """
        return query

    def on_after_invoke(self, query: str, response: str) -> str:
        """
        Agent invoke 后调用，可修改 response。

        用于后处理模型输出，例如格式化、审核等。

        Args:
            query: 原始用户输入
            response: Agent 输出

        Returns:
            （可能修改过的）Agent 输出
        """
        return response

    # ==================== 属性 ====================

    @property
    def name(self) -> str:
        """Skill 名称"""
        return self.config.name

    @property
    def is_active(self) -> bool:
        """Skill 是否处于激活状态"""
        return self._is_active

    @property
    def tags(self) -> List[str]:
        """Skill 标签"""
        return self.config.tags

    @property
    def priority(self) -> int:
        """Skill 优先级"""
        return self.config.priority

    def get_tool_names(self) -> List[str]:
        """获取此 Skill 提供的所有工具名称"""
        return [tool.name for tool in self.get_tools()]

    def build_manifest(self) -> SkillManifest:
        """构建此 Skill 的统一元数据。"""
        return SkillManifest(
            name=self.name,
            description=self.config.description,
            listing_description=self.get_listing_description(),
            when_to_use=self.get_when_to_use(),
            version=self.config.version,
            tags=list(self.config.tags),
            priority=self.config.priority,
            exposure_mode=self.get_exposure_mode(),
            execution_mode=self.get_execution_mode(),
            source_type=self.config.source_type,
            source_path=self.config.source_path,
            cache_lifecycle=self.get_cache_lifecycle(),
            tool_names=self.get_tool_names(),
            metadata=dict(self.config.extra),
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "description": self.config.description,
            "listing_description": self.get_listing_description(),
            "when_to_use": self.get_when_to_use(),
            "version": self.config.version,
            "tags": self.config.tags,
            "priority": self.config.priority,
            "exposure_mode": self.get_exposure_mode(),
            "execution_mode": self.get_execution_mode(),
            "cache_lifecycle": self.get_cache_lifecycle(),
            "is_active": self._is_active,
            "tools": self.get_tool_names(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"active={self._is_active}, "
            f"tools={self.get_tool_names()})"
        )
