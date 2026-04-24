"""
TaskPlanningSkill — 结构化任务规划技能

封装 TaskCreate / Get / Update / List，用于把长任务拆成可跟踪对象。
"""
from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig
from task import InMemoryTaskStore, TaskService

if TYPE_CHECKING:
    from Tool.BaseTool import Tool

logger = logging.getLogger(__name__)


class TaskPlanningSkill(BaseSkill):
    """
    任务规划技能

    适合把复杂目标拆成结构化任务，并在执行过程中持续更新状态。
    """

    def __init__(self, service: Optional[TaskService] = None):
        config = SkillConfig(
            name="task_planning",
            description="结构化任务规划技能，支持创建、读取、更新和列出任务。",
            listing_description="Break large goals into trackable structured tasks.",
            when_to_use="当任务较长、涉及多个步骤或需要持续追踪状态时使用。",
            version="1.0.0",
            tags=["task", "planning", "execution", "tracking"],
            priority=6,
        )
        super().__init__(config)
        self.service = service or TaskService(InMemoryTaskStore())

    def get_tools(self) -> List["Tool"]:
        """返回结构化任务工具。"""
        from Tool.builtin import TaskCreateTool, TaskGetTool, TaskListTool, TaskUpdateTool

        return [
            TaskCreateTool(service=self.service),
            TaskGetTool(service=self.service),
            TaskUpdateTool(service=self.service),
            TaskListTool(service=self.service),
        ]

    def get_prompt(self) -> str:
        """返回任务规划使用指南。"""
        return """## 结构化任务规划能力
你可以把复杂目标拆成结构化任务，并在执行过程中维护状态。

建议做法：
- 面对多步骤任务时，先用 `TaskCreate` 创建任务对象，而不是只在脑中维护计划
- 拆分粒度要适中：每个任务都应有明确产出或验证标准
- 执行前先 `TaskList` / `TaskGet`，避免重复创建
- 进展变化时及时 `TaskUpdate`，保持状态和 owner 信息最新

规划原则：
- 任务标题简短清晰，描述里写清目标、边界和验收条件
- 长任务要拆分为可独立完成的小任务
- 最好保留验证或回归测试类任务，不要只规划实现任务
"""
