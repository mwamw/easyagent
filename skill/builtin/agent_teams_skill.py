"""
AgentTeamsSkill — 多 Agent 团队协作技能

沉淀 plan -> team -> synthesize 的团队协作模式。
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool


class AgentTeamsSkill(BaseSkill):
    """
    多 Agent 团队协作技能

    适合在已有 Agent / Team / Mailbox 运行时工具的前提下指导团队编排。
    """

    def __init__(self):
        config = SkillConfig(
            name="agent_teams",
            description="多 Agent 团队协作技能，帮助模型以 plan -> team -> synthesize 方式组织子 agent。",
            listing_description="Plan, spawn, coordinate, and synthesize multi-agent teamwork.",
            when_to_use="当任务可并行拆分，且 runtime 已提供 Agent、TeamCreate、SendMessage、AgentWait 等工具时使用。",
            version="1.0.0",
            tags=["multi-agent", "team", "runtime", "orchestration", "mailbox"],
            priority=9,
        )
        super().__init__(config)

    def get_tools(self) -> List["Tool"]:
        return []

    def get_prompt(self) -> str:
        return """## 多 Agent 团队协作能力
在具备团队运行时工具时，优先采用 `plan -> team -> synthesize` 的协作流程。

推荐流程：
1. Plan phase：先写团队 brief，明确要不要拆分、拆成几个 agent、各自边界、依赖顺序和验收标准
2. Team phase：只把可并行且边界清晰的任务交给子 agent，避免多人编辑同一组文件
3. Message phase：通过团队广播补充统一约束，例如输出格式、禁止事项、优先级变化
4. Synthesize phase：不要只看“已启动”，要收集每个 agent 的状态、outputFile、最终结论再汇总

编排原则：
- 子 agent 的职责边界必须具体到模块、目录或文件范围
- 阻塞路径上的结果，不要无限后台化；必要时显式等待
- 后台 agent 启动后要检查是否完成，不要只创建不回收
- 任务变更时优先用结构化消息同步，而不是假设对方会自己推断
"""
