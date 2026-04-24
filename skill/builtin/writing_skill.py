"""
WritingSkill — 通用写作技能

提升结构、语气、清晰度和信息密度，不依赖额外工具。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class WritingSkill(BaseSkill):
    """
    通用写作技能

    适合说明文、邮件、文档、博客、公告和长短文本重写。
    """

    def __init__(self):
        config = SkillConfig(
            name="writing",
            description="通用写作技能，帮助模型提升结构、表达、语气控制和可读性。",
            listing_description="Write clearer, sharper, and better-structured text.",
            when_to_use="当你要写说明文、邮件、公告、博客、文档、方案说明或改写现有文本时使用。",
            version="1.0.0",
            tags=["writing", "communication", "editing", "documentation"],
            priority=7,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 通用写作能力
写作时要优先保证信息清晰、结构稳定、句子有节奏，而不是堆砌术语或空话。

写作原则：
- 先明确目标读者、写作目的和希望对方得到什么结论
- 开头尽快进入主题，不要拖很长前言
- 每一段都应承载一个清晰功能：定义问题、展开原因、说明方案、给出结论
- 句子尽量直接，删除低信息密度修饰词和重复表达

质量要求：
- 结构清楚：读者能迅速知道重点和次重点
- 语气合适：专业但不僵硬，明确但不过度强势
- 信息密度高：少空泛总结，多给具体事实、例子、条件和边界
- 可扫读：标题、段落、列表、术语命名都应便于快速浏览

改写要求：
- 改写时先保留原意，再优化结构和语气
- 如果原文逻辑混乱，要直接重组，不要只做表层润色
- 如果需要压缩篇幅，优先删重复和废话，不要删掉关键限定条件
"""
