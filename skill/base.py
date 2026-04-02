"""
Skill 基类模块

定义 Skill 的抽象接口和配置数据结构。
Skill 是一个「能力包」，将 Tools + Prompt + ContextSource + 配置封装为独立单元。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from Tool.BaseTool import Tool
    from context.source.base import BaseContextSource


class SkillConfig(BaseModel):
    """Skill 配置数据类"""

    name: str = Field(description="Skill 唯一标识名称")
    description: str = Field(default="", description="Skill 功能描述")
    version: str = Field(default="1.0.0", description="Skill 版本号")
    tags: List[str] = Field(default_factory=list, description="标签（用于搜索/分类）")
    priority: int = Field(default=0, description="优先级，数值越大越靠前注入 prompt")
    auto_activate: bool = Field(default=True, description="注册到 SkillManager 时是否自动激活")
    dependencies: List[str] = Field(default_factory=list, description="依赖的其他 Skill 名称列表")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Skill 自定义扩展配置")


class BaseSkill(ABC):
    """
    Skill 抽象基类

    所有 Skill 实现应继承此类。Skill 封装了一组相关的能力，
    包括工具、prompt 指导、上下文来源和生命周期钩子。

    典型子类只需实现 get_tools() 和 get_prompt() 两个核心方法。

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
        返回此 Skill 的 system prompt 片段。

        该片段会被拼接到 Agent 的 system prompt 中，指导 LLM 如何使用此 Skill 的能力。

        Returns:
            prompt 文本片段
        """
        ...

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

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "description": self.config.description,
            "version": self.config.version,
            "tags": self.config.tags,
            "priority": self.config.priority,
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
