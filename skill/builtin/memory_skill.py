"""
MemorySkill — 记忆系统技能

将 V2 记忆系统（MemoryManage）的全部能力封装为 Skill，包括：
- 6 个记忆工具（增删改查搜索维护）
- 记忆系统使用指南 prompt
- Working Memory 便签本实时注入
- MemoryContextSource 上下文来源
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from memory.V2.MemoryManage import MemoryManage
    from Tool.BaseTool import Tool
    from context.source.base import BaseContextSource

logger = logging.getLogger(__name__)


class MemorySkill(BaseSkill):
    """
    记忆系统技能 — 封装 V2 记忆系统的全部能力

    功能:
    - 提供 add/search/get/update/remove/maintenance 6 个记忆操作工具
    - 提供记忆系统使用指南 prompt（含 Working Memory 便签本内容注入）
    - 可选提供 MemoryContextSource 供 ContextManager 使用

    Example::

        from memory.V2.MemoryManage import MemoryManage
        from skill.builtin.memory_skill import MemorySkill

        mm = MemoryManage(config, user_id="user1", ...)
        skill = MemorySkill(memory_manage=mm)
        agent.with_skill(skill)
    """

    def __init__(
        self,
        memory_manage: "MemoryManage",
        session_id: Optional[str] = None,
        include_context_source: bool = True,
    ):
        """
        初始化 MemorySkill

        Args:
            memory_manage: V2 记忆管理器实例
            session_id: 会话 ID（不提供则自动生成）
            include_context_source: 是否包含 MemoryContextSource
        """
        config = SkillConfig(
            name="memory",
            description="V2 多层记忆系统技能，支持工作记忆、情景记忆、语义记忆和感知记忆的读写检索",
            version="2.0.0",
            tags=["memory", "knowledge", "recall", "working_memory"],
            priority=10,  # 高优先级，prompt 排在前面
        )
        super().__init__(config)

        self.memory_manage = memory_manage
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._include_context_source = include_context_source

    def get_tools(self) -> List["Tool"]:
        """返回全部 6 个 V2 记忆工具"""
        from Tool.builtin.memorytool import (
            AddMemoryTool,
            SearchMemoryTool,
            GetMemoryTool,
            UpdateMemoryTool,
            RemoveMemoryTool,
            MemoryMaintenanceTool,
        )

        return [
            AddMemoryTool(self.memory_manage, current_session_id=self.session_id),
            SearchMemoryTool(self.memory_manage, current_session_id=self.session_id),
            GetMemoryTool(self.memory_manage),
            UpdateMemoryTool(self.memory_manage),
            RemoveMemoryTool(self.memory_manage),
            MemoryMaintenanceTool(self.memory_manage),
        ]

    def get_prompt(self) -> str:
        """返回记忆系统使用指南 prompt + Working Memory 便签本"""
        supported = self.memory_manage.get_supported_type()
        prompt = f"""## 你的记忆系统 (The Memory System)
你拥有一个高级记忆管理系统（当前支持: {supported}），能够跨越长期和短期存储知识。
- **必要性原则**：仅当当前对话上下文（History）不足以回答问题时，才考虑使用搜索工具。禁止对显而易见或刚讨论过的信息进行重复搜索。
- **工作记忆 (Working Memory)**：遇到关键的约束、当前任务大纲或中间状态，请主动调用工具写入。这是你的"即时贴"，用于存放当前任务最核心的信息。
- **长期记忆搜索 (Search)**：遇到不知道的事实或历史脉络，请使用 search_memory_tool 到 semantic（语义）或 episodic（情景）记忆中检索。
- **记忆持久化**：只有你主动存入，未来的你才能回想起现在的经历和设定。
- **按需使用**：其他记忆工具（如删除、更新）仅在信息过时或用户要求时使用。
"""
        # 注入 Working Memory 便签本
        if "working" in self.memory_manage.memory_types:
            try:
                working_memories = self.memory_manage.memory_types["working"].get_all_memories()
                if working_memories:
                    wm_texts = [f"- id:{m.id}: {m.content}" for m in working_memories]
                    wm_str = "\n".join(wm_texts)
                    prompt += f"\n\n【当前工作便签本（Working Memory）】:\n{wm_str}"
                else:
                    prompt += "\n\n【当前工作便签本（Working Memory）】:\n(空)"
            except Exception as e:
                logger.warning("读取 Working Memory 失败: %s", e)
                prompt += "\n\n【当前工作便签本（Working Memory）】:\n(读取失败)"

            prompt += "\n(注: 遇到复杂任务时，请主动调用 add_memory_tool 记录约束条件和中间结论)"
            prompt += "\n(注: 当任务结束或话题转变时，务必主动调用 memory_maintenance_tool 清理无用便签或用 remove_memory_tool 删除指定id内容)\n"

        return prompt

    def get_context_sources(self) -> List["BaseContextSource"]:
        """返回 MemoryContextSource（如果启用）"""
        if not self._include_context_source:
            return []
        try:
            from context.source.memory_source import MemoryContextSource
            return [MemoryContextSource(memory_manage=self.memory_manage)]
        except ImportError:
            logger.warning("无法导入 MemoryContextSource")
            return []

    def on_activate(self, agent: Any) -> None:
        """激活时记录到 Agent"""
        logger.info("🧠 MemorySkill 已激活 (session=%s)", self.session_id)

    def on_deactivate(self, agent: Any) -> None:
        """停用时记录"""
        logger.info("🧠 MemorySkill 已停用")
