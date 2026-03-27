"""
多 Agent 协作 — 顺序编排器 (Sequential Pipeline)

SequentialOrchestrator 按照预定义的流水线顺序依次调用 Agent，
每个 Agent 的输出作为下一个 Agent 的输入。

典型用例：
    researcher → writer → reviewer

用法示例::

    orchestrator = SequentialOrchestrator(
        name="报告生成流水线",
        pipeline=["researcher", "writer", "reviewer"],
    )
    orchestrator.add_agent("researcher", researcher_agent)
    orchestrator.add_agent("writer", writer_agent)
    orchestrator.add_agent("reviewer", reviewer_agent)
    result = orchestrator.run("写一篇关于 AI Agent 的调研报告")
"""
from __future__ import annotations

import time
import logging
from typing import Optional, TYPE_CHECKING

from .base import BaseOrchestrator
from .context import SharedContext
from .exceptions import OrchestrationError

if TYPE_CHECKING:
    from core.callbacks import CallbackManager

logger = logging.getLogger(__name__)


class SequentialOrchestrator(BaseOrchestrator):
    """
    顺序流水线编排器
    
    Agent 按 pipeline 定义的顺序依次执行，
    每个 Agent 的输出作为下一个 Agent 的输入。
    
    如果某个 Agent 失败：
    - continue_on_error=False（默认）：立即中止，抛出异常
    - continue_on_error=True：跳过失败 Agent，将前一个成功结果传递给后续 Agent
    
    Attributes:
        pipeline: Agent 执行顺序列表
        continue_on_error: 是否在某个 Agent 失败时继续执行后续 Agent
    """

    def __init__(
        self,
        name: str,
        pipeline: list[str],
        continue_on_error: bool = False,
        temperature: float = 0.7,
        max_iter_per_agent: int = 10,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        super().__init__(name=name, callback_manager=callback_manager)
        self.pipeline = pipeline
        self.continue_on_error = continue_on_error
        self._temperature = temperature
        self._max_iter_per_agent = max_iter_per_agent

        logger.info(
            "🔗 SequentialOrchestrator '%s' 初始化: pipeline=%s, continue_on_error=%s",
            name, pipeline, continue_on_error,
        )

    def run(self, query: str, **kwargs) -> str:
        """
        按 pipeline 顺序执行 Agent 流水线
        
        Args:
            query: 用户原始输入
            
        Returns:
            流水线最后一个成功 Agent 的输出
            
        Raises:
            OrchestrationError: 流水线执行失败
        """
        # 校验
        self._validate_agents(required_names=self.pipeline)

        context = SharedContext(original_query=query)
        total_start = time.time()

        logger.info(
            "🚀 编排器 '%s' 开始执行: query='%s' pipeline=%s",
            self.name, query[:50], self.pipeline,
        )

        current_input = query
        completed_agents: list[str] = []
        failed_agents: list[str] = []

        for step_idx, agent_name in enumerate(self.pipeline, start=1):
            logger.info(
                "--- 步骤 %d/%d: Agent '%s' ---",
                step_idx, len(self.pipeline), agent_name,
            )

            # 构造传递给当前 Agent 的 prompt
            # 首个 Agent 直接用原始 query
            # 后续 Agent 收到上一个 Agent 的输出 + 上下文
            if step_idx == 1:
                agent_input = current_input
            else:
                prev_agent = completed_agents[-1] if completed_agents else "user"
                agent_input = self._build_handoff_prompt(
                    original_query=query,
                    prev_agent=prev_agent,
                    prev_output=current_input,
                    current_agent=agent_name,
                    step_idx=step_idx,
                    total_steps=len(self.pipeline),
                )

            # 触发 handoff 回调
            if step_idx > 1 and completed_agents:
                self._trigger_handoff(
                    from_agent=completed_agents[-1],
                    to_agent=agent_name,
                    task=agent_input[:100],
                )

            try:
                result_msg = self._invoke_agent(
                    agent_name=agent_name,
                    query=agent_input,
                    context=context,
                    temperature=self._temperature,
                    max_iter=self._max_iter_per_agent,
                )
                current_input = result_msg.content
                completed_agents.append(agent_name)

            except OrchestrationError as e:
                failed_agents.append(agent_name)
                if self.continue_on_error:
                    logger.warning(
                        "⚠️  Agent '%s' 失败，continue_on_error=True，跳过继续: %s",
                        agent_name, e
                    )
                    # 保持 current_input 不变，传递给下一个 Agent
                    continue
                else:
                    logger.error(
                        "💥 Agent '%s' 失败，中止流水线: %s",
                        agent_name, e
                    )
                    raise

        total_duration_ms = int((time.time() - total_start) * 1000)

        logger.info(
            "🏁 编排器 '%s' 执行完成: 总耗时=%dms, 完成=%s, 失败=%s, 输出长度=%d",
            self.name, total_duration_ms,
            completed_agents, failed_agents, len(current_input),
        )

        # 触发编排器结束回调
        self._trigger_orchestrator_end(
            result=current_input,
            rounds=len(self.pipeline),
        )

        return current_input

    def _build_handoff_prompt(
        self,
        original_query: str,
        prev_agent: str,
        prev_output: str,
        current_agent: str,
        step_idx: int,
        total_steps: int,
    ) -> str:
        """
        构造 Agent 间流转的 Prompt
        
        让当前 Agent 知道：
        1. 原始用户需求是什么
        2. 前一个 Agent 的输出是什么
        3. 自己在流水线中的位置
        """
        return (
            f"你正在参与一个多步骤协作流水线（步骤 {step_idx}/{total_steps}）。\n"
            f"\n"
            f"【用户原始需求】\n{original_query}\n"
            f"\n"
            f"【上一步 Agent '{prev_agent}' 的输出】\n{prev_output}\n"
            f"\n"
            f"请基于以上信息，完成你作为 '{current_agent}' 的职责。"
        )
