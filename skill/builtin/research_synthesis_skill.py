"""
ResearchSynthesisSkill — 研究综述与资料综合技能

帮助模型把多来源信息整理为结构化结论。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class ResearchSynthesisSkill(BaseSkill):
    """
    研究综述技能

    适合资料整理、比较分析、结论归纳与证据分层。
    """

    def __init__(self):
        config = SkillConfig(
            name="research_synthesis",
            description="研究综述与资料综合技能，帮助模型将多来源信息收敛为结构化结论。",
            listing_description="Synthesize many sources into a clear, evidence-aware summary.",
            when_to_use="当你已经拿到多份资料，需要做综述、比较、证据分层、结论提炼或输出研究报告时使用。",
            version="1.0.0",
            tags=["research", "analysis", "synthesis", "comparison", "evidence"],
            priority=8,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 研究综述与资料综合能力
面对多来源资料时，不要按时间顺序复述，而要主动抽象、比较和归纳。

工作流：
- 先识别每份材料的类型：原始事实、二手解释、观点判断、推测
- 再抽取共识、分歧、缺口和高不确定性区域
- 最后用统一结构输出，而不是把来源逐个摘要堆起来

综合原则：
- 区分“事实”“解释”“结论”“推断”
- 来源不一致时，明确冲突点和可能原因，不要强行拼成一个结论
- 证据强的内容前置，弱证据或推测显式降权
- 如果材料很多，优先按主题、变量、时间线或立场分组

输出要求：
- 先给高层结论，再展开支撑证据
- 比较时使用同一维度，不要每段都换标准
- 如果存在未知项，直接指出还缺什么信息，而不是用模糊语气掩盖
"""
