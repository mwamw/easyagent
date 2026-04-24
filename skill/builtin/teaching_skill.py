"""
TeachingSkill — 教学解释技能

帮助模型把复杂内容讲清楚，并兼顾层次与例子。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class TeachingSkill(BaseSkill):
    """
    教学解释技能

    适合讲概念、带练习、做分层解释与建立直觉。
    """

    def __init__(self):
        config = SkillConfig(
            name="teaching",
            description="教学解释技能，帮助模型分层、循序渐进地讲清复杂概念。",
            listing_description="Explain complex ideas clearly with layered teaching and examples.",
            when_to_use="当你需要解释复杂概念、做教学答疑、写教程、设计练习或给初学者建立直觉时使用。",
            version="1.0.0",
            tags=["teaching", "education", "explanation", "tutorial"],
            priority=7,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 教学解释能力
教学时要让对方真正理解，而不是只把定义换个说法重复一遍。

讲解原则：
- 先判断学习者最可能卡在哪里，再安排解释顺序
- 先建立直觉，再讲定义，再讲边界和例外
- 抽象概念尽量配具体例子、反例和类比
- 如果内容复杂，分层讲：一句话版、直觉版、正式版、进阶版

高质量讲解应包含：
- 这个概念解决什么问题
- 它和相近概念有什么区别
- 什么时候适用，什么时候不适用
- 初学者最容易犯什么错

输出要求：
- 不要一上来堆术语
- 不要默认听众已经知道隐含前提
- 如果用户只问一个点，不要无节制扩展到整套课程
- 如果合适，可主动给小练习或检验理解的问题
"""
