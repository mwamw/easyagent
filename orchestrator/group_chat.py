"""
多 Agent 协作 — 群聊编排器 (Group Chat)

GroupChatOrchestrator 让多个 Agent 在共享上下文中轮流发言讨论。
支持两种发言人选择策略：
- round_robin: 按固定顺序轮流
- moderator: 由一个 Moderator Agent 决定下一个发言者

典型用例：技术评审会、头脑风暴、辩论

用法示例::

    orchestrator = GroupChatOrchestrator(
        name="技术评审会",
        moderator_llm=llm,
        max_rounds=5,
        speaker_selection="moderator",
    )
    orchestrator.add_agent("architect", architect_agent)
    orchestrator.add_agent("security_expert", security_agent)
    orchestrator.add_agent("tester", tester_agent)
    result = orchestrator.run("评审这个微服务架构方案")
"""
from __future__ import annotations

import time
import logging
from typing import Optional, Literal, TYPE_CHECKING

from .base import BaseOrchestrator
from .context import SharedContext
from .exceptions import OrchestrationError, MaxRoundsExceededError

if TYPE_CHECKING:
    from core.llm import EasyLLM
    from core.callbacks import CallbackManager

logger = logging.getLogger(__name__)

SpeakerSelection = Literal["round_robin", "moderator"]


class GroupChatOrchestrator(BaseOrchestrator):
    """
    群聊编排器
    
    多个 Agent 在共享上下文中轮流发言。
    每个 Agent 都能看到完整的对话历史。
    
    发言人选择策略：
    - "round_robin": 按注册顺序轮流发言
    - "moderator": 由 Moderator LLM 分析对话后选择下一个发言者
    
    结束条件（任一满足）：
    - 达到 max_rounds 轮
    - Moderator 判断讨论已充分（moderator 模式）
    
    Attributes:
        moderator_llm: 主持人使用的 LLM (用于选择下一个发言者和最终总结)
        max_rounds: 最大讨论轮数
        speaker_selection: 发言人选择策略
    """

    def __init__(
        self,
        name: str,
        moderator_llm: "EasyLLM",
        max_rounds: int = 5,
        speaker_selection: SpeakerSelection = "round_robin",
        temperature: float = 0.7,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        super().__init__(name=name, callback_manager=callback_manager)
        self._moderator_llm = moderator_llm
        self._max_rounds = max_rounds
        self._speaker_selection = speaker_selection
        self._temperature = temperature

        logger.info(
            "💬 GroupChatOrchestrator '%s' 初始化: max_rounds=%d, selection=%s",
            name, max_rounds, speaker_selection,
        )

    def run(self, query: str, **kwargs) -> str:
        """
        启动群聊讨论
        
        Args:
            query: 讨论议题
            
        Returns:
            Moderator 的最终总结
        """
        self._validate_agents()

        context = SharedContext(original_query=query)
        total_start = time.time()
        agent_names = self.agent_names

        logger.info(
            "🚀 群聊编排器 '%s' 开始: 议题='%s', 参与者=%s, 最大轮次=%d",
            self.name, query[:50], agent_names, self._max_rounds,
        )

        # 记录初始议题
        context.add(
            sender="user",
            receiver="all",
            content=query,
            msg_type="task",
        )

        should_conclude = False

        for round_idx in range(1, self._max_rounds + 1):
            logger.info(
                "=== 群聊第 %d/%d 轮 ===",
                round_idx, self._max_rounds,
            )

            # 选择发言人列表
            if self._speaker_selection == "round_robin":
                speakers = agent_names
            else:
                # moderator 模式：让 Moderator 决定这轮谁发言
                speakers, should_conclude = self._select_speakers(
                    context, round_idx
                )
                if should_conclude:
                    logger.info("🔔 Moderator 判断讨论可以结束")
                    break

            # 每个 speaker 依次发言
            for speaker_name in speakers:
                speaker_prompt = self._build_speaker_prompt(
                    speaker_name, context, round_idx
                )

                self._trigger_handoff("moderator", speaker_name, f"轮次{round_idx}发言")

                try:
                    self._invoke_agent(
                        agent_name=speaker_name,
                        query=speaker_prompt,
                        context=context,
                        temperature=self._temperature,
                    )
                except OrchestrationError as e:
                    logger.warning(
                        "⚠️  Agent '%s' 在第 %d 轮发言失败: %s，继续讨论",
                        speaker_name, round_idx, e,
                    )
                    # 群聊中单个 Agent 失败不中止整场讨论

        # 最终总结
        summary = self._generate_summary(context)

        total_duration_ms = int((time.time() - total_start) * 1000)

        logger.info(
            "🏁 群聊编排器 '%s' 完成: 轮次=%d, 耗时=%dms, 总结长度=%d",
            self.name,
            min(round_idx, self._max_rounds),  # type: ignore
            total_duration_ms,
            len(summary),
        )

        self._trigger_orchestrator_end(summary, round_idx)  # type: ignore
        return summary

    # ==================== 发言人选择 ====================

    def _select_speakers(
        self, context: SharedContext, round_idx: int
    ) -> tuple[list[str], bool]:
        """
        让 Moderator LLM 选择下一轮的发言者
        
        Returns:
            (speaker_names, should_conclude)
        """
        transcript = context.get_full_transcript()
        agent_names = self.agent_names

        prompt = (
            f"你是一个群聊讨论的主持人。\n"
            f"\n"
            f"【讨论议题】\n{context.original_query}\n"
            f"\n"
            f"【已有讨论记录】\n{transcript}\n"
            f"\n"
            f"【可选发言者】\n{', '.join(agent_names)}\n"
            f"\n"
            f"当前是第 {round_idx} 轮讨论。请决定：\n"
            f"1. 如果讨论已经足够充分，请回复: CONCLUDE\n"
            f"2. 否则，请回复下一轮应该发言的参与者名称（用逗号分隔），例如: architect,tester\n"
            f"\n"
            f"只回复名称或 CONCLUDE，不要其他内容。"
        )

        try:
            response = self._moderator_llm.invoke(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            response_text = (response or "").strip()

            logger.debug("Moderator 选择: %s", response_text)

            if "CONCLUDE" in response_text.upper():
                return [], True

            # 解析发言者名称
            selected = [
                name.strip() for name in response_text.split(",")
                if name.strip() in agent_names
            ]

            if not selected:
                logger.warning(
                    "⚠️  Moderator 选择了无效的发言者 '%s'，使用 round_robin",
                    response_text,
                )
                return agent_names, False

            return selected, False

        except Exception as e:
            logger.warning("⚠️  Moderator 选择失败: %s，使用 round_robin", e)
            return agent_names, False

    # ==================== Prompt 构建 ====================

    def _build_speaker_prompt(
        self, speaker_name: str, context: SharedContext, round_idx: int
    ) -> str:
        """构建给发言者的 prompt：包含讨论议题和之前的对话"""
        transcript = context.get_full_transcript()

        return (
            f"你正在参与一场多人讨论（第 {round_idx} 轮）。\n"
            f"\n"
            f"【讨论议题】\n{context.original_query}\n"
            f"\n"
            f"【之前的讨论记录】\n{transcript}\n"
            f"\n"
            f"现在轮到你 ('{speaker_name}') 发言。\n"
            f"请基于你的专业角色给出观点、补充、质疑或建议。\n"
            f"发言要简洁有力，避免重复已说过的内容。"
        )

    def _generate_summary(self, context: SharedContext) -> str:
        """让 Moderator 生成最终总结"""
        transcript = context.get_full_transcript()

        prompt = (
            f"你是讨论的主持人，请对以下讨论做最终总结。\n"
            f"\n"
            f"【讨论议题】\n{context.original_query}\n"
            f"\n"
            f"【完整讨论记录】\n{transcript}\n"
            f"\n"
            f"请总结讨论的：\n"
            f"1. 主要观点和共识\n"
            f"2. 存在的分歧\n"
            f"3. 最终建议或结论\n"
        )

        try:
            summary = self._moderator_llm.invoke(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return summary or "讨论总结生成失败。"
        except Exception as e:
            logger.error("❌ 生成讨论总结失败: %s", e)
            # 兜底：返回原始记录
            return f"讨论总结生成失败。原始记录:\n{transcript}"
