"""
DebugSkill — 系统化调试技能

基于假设驱动和证据优先的调试流程。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class DebugSkill(BaseSkill):
    """
    调试技能

    强调缩小问题空间、保留证据链、避免拍脑袋修改。
    """

    def __init__(self):
        config = SkillConfig(
            name="debug",
            description="系统化调试技能，强调假设驱动、证据优先和逐步收敛。",
            listing_description="Hypothesis-driven debugging workflow for bugs and failures.",
            when_to_use="当你在排查 bug、测试失败、运行时异常、构建错误或线上问题线索时使用。",
            version="1.0.0",
            tags=["debug", "bugfix", "failure-analysis", "verification"],
            priority=8,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 系统化调试能力
调试时要以证据驱动，而不是凭直觉连续乱改。

工作流：
- 先复现：确认现象、输入条件、触发步骤、错误信息
- 再收集证据：日志、测试输出、调用链、最近变更、相关配置
- 提出少量可验证假设，并逐个排除
- 每次只做最小修改，然后立即验证结果

调试要求：
- 不要在根因未明时同时改多个独立位置
- 不要把“错误消失了”直接当成“根因找到了”
- 修复后要验证主路径和回归场景，必要时补测试
- 如果暂时无法彻底修复，要明确记录现象、推断、剩余风险和下一步
"""
