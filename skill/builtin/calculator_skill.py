"""
CalculatorSkill — 计算器技能

封装计算器工具和使用指南。
"""
from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool

logger = logging.getLogger(__name__)


class CalculatorSkill(BaseSkill):
    """
    计算器技能

    提供数学计算能力，适用于需要精确数值计算的场景。

    Example::

        skill = CalculatorSkill()
        agent.with_skill(skill)
    """

    def __init__(self):
        config = SkillConfig(
            name="calculator",
            description="数学计算技能，支持各种数学运算",
            version="1.0.0",
            tags=["math", "calculation", "compute"],
            priority=3,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        """返回计算器工具"""
        from Tool.builtin.calculator import CalculatorTool
        return [CalculatorTool()]

    def get_prompt(self) -> str:
        """返回计算器使用指南"""
        return """## 数学计算能力
你具备精确的数学计算能力。当遇到以下情况时，请使用计算器工具：
- 复杂的数学运算（大数乘除、幂运算、开方等）
- 统计计算（平均值、标准差等）
- 单位换算或比例计算
- 任何需要精确数值结果的场景
注意：简单的加减运算可以直接心算回答，无需调用工具。
"""
