"""
WebSearchSkill — Web 搜索技能

封装 Web 搜索工具和使用指南。
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool

logger = logging.getLogger(__name__)


class WebSearchSkill(BaseSkill):
    """
    Web 搜索技能

    提供联网搜索能力，适用于需要查询实时信息、新闻或事实验证的场景。

    Example::

        skill = WebSearchSkill(api_key="your-api-key")
        agent.with_skill(skill)
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        初始化 WebSearchSkill

        Args:
            api_key: 搜索 API 密钥
            **kwargs: 传递给 WebSearchTool 的额外参数
        """
        config = SkillConfig(
            name="web_search",
            description="Web 搜索技能，支持联网搜索和信息检索",
            version="1.0.0",
            tags=["search", "web", "real-time", "information"],
            priority=5,
        )
        super().__init__(config)
        self._api_key = api_key
        self._kwargs = kwargs

    def get_tools(self) -> List["Tool"]:
        """返回 Web 搜索工具"""
        from Tool.builtin.search import WebSearchTool
        return [WebSearchTool(api_key=self._api_key, **self._kwargs)]

    def get_prompt(self) -> str:
        """返回搜索使用指南"""
        return """## Web 搜索能力
你具备联网搜索信息的能力。请遵循以下原则：
- 当用户询问**实时信息**（天气、新闻、股票、时事等）时，主动使用搜索工具
- 当需要**验证事实**或查找**最新数据**时，使用搜索工具
- 搜索后请**综合多个来源**给出可靠回答，注明信息来源
- 对于已知的常识性问题，**无需搜索**，直接回答
- 搜索关键词要**精准简洁**，避免冗长的查询语句
"""
