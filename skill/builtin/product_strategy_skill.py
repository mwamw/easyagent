"""
ProductStrategySkill — 产品策略技能

帮助模型以产品视角定义问题、用户、约束和取舍。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class ProductStrategySkill(BaseSkill):
    """
    产品策略技能

    适合需求分析、方案比较、范围取舍和路线图讨论。
    """

    def __init__(self):
        config = SkillConfig(
            name="product_strategy",
            description="产品策略技能，帮助模型从用户价值、业务目标、约束与取舍角度分析问题。",
            listing_description="Think in product terms: users, value, constraints, and tradeoffs.",
            when_to_use="当你在分析需求、定义 MVP、比较方案、排优先级、写 PRD 或讨论路线图时使用。",
            version="1.0.0",
            tags=["product", "strategy", "roadmap", "requirements", "prioritization"],
            priority=8,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 产品策略能力
你需要从产品角度思考，而不是只从功能角度罗列需求。

分析框架：
- 用户是谁，痛点是什么，当前替代方案是什么
- 这个功能解决的核心问题是什么，什么不是它要解决的问题
- 哪些目标是必须达成的，哪些只是理想状态
- 资源、时间、实现成本、组织依赖分别带来什么约束

决策原则：
- 先抓核心价值，再扩边界，不要一开始就追求“功能全”
- 方案比较必须明确 tradeoff，而不是都说“各有优劣”
- 优先级应同时考虑用户价值、业务价值、实现成本和风险
- MVP 要能独立成立，不是把完整产品随便砍几刀

输出要求：
- 明确问题定义、用户对象、成功指标和边界
- 如果给方案，至少说明为什么选它、不选什么、代价是什么
- 如果信息不足，要指出关键未知项，而不是假装能精确决策
"""
