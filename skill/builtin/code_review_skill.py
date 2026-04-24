"""
CodeReviewSkill — 代码审查技能

基于社区常见的结构化 review workflow，提供面向 correctness/security/performance/tests
的审查方法。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class CodeReviewSkill(BaseSkill):
    """
    代码审查技能

    侧重结构化 review，而不是零散风格建议。
    """

    def __init__(self):
        config = SkillConfig(
            name="code_review",
            description="结构化代码审查技能，帮助模型按正确性、架构、安全、性能和测试维度审查改动。",
            listing_description="Structured code review workflow with severity-based findings.",
            when_to_use="当你需要 review PR、审查自己或他人写的代码、或在合并前做质量把关时使用。",
            version="1.0.0",
            tags=["review", "quality", "security", "performance", "testing"],
            priority=8,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 代码审查能力
你应按结构化 review 流程审查改动，而不是给出零散、模糊的建议。

推荐流程：
1. 先收集上下文：变更范围、目标、约束、相关文件和测试
2. 再做高层审查：架构、接口边界、状态流、回归风险
3. 最后做细节审查：逻辑正确性、安全问题、性能问题、边界条件、可维护性

审查维度：
- Correctness：实现是否真的满足需求，异常路径和边界是否处理
- Readability：命名、结构、控制流是否清晰，是否引入不必要复杂度
- Architecture：模块边界、依赖方向、抽象层次是否合理
- Security：输入校验、权限、数据暴露、注入风险、敏感信息处理
- Performance：无谓 IO、重复计算、N+1、缓存缺失、阻塞路径
- Tests：是否覆盖核心路径、错误路径和回归场景

输出要求：
- 以 findings 为主，按严重程度排序
- 每条问题都要尽量给出证据、影响和触发条件
- 风格建议排在行为错误之后，不要把格式问题当主问题
"""
