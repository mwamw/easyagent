"""
多 Agent 协作 — Supervisor 编排器

SupervisorOrchestrator 使用一个 Supervisor Agent 来动态决策
应该调用哪个 Worker Agent 来处理当前任务。

Supervisor 通过工具调用来委派任务：
- handoff_to_agent(agent_name, task): 将子任务交给指定 Worker
- final_answer(answer): 给出最终答案，结束编排

用法示例::

    orchestrator = SupervisorOrchestrator(
        name="智能助手",
        supervisor_llm=llm,
        max_rounds=10,
    )
    orchestrator.add_agent("coder", coder_agent)
    orchestrator.add_agent("researcher", researcher_agent)
    result = orchestrator.run("帮我写一个快速排序并解释原理")
"""
from __future__ import annotations

import json
import time
import logging
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from .base import BaseOrchestrator
from .message import AgentMessage
from .context import SharedContext
from .exceptions import (
    OrchestrationError,
    MaxRoundsExceededError,
    HandoffError,
)
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry

if TYPE_CHECKING:
    from core.llm import EasyLLM
    from core.callbacks import CallbackManager

logger = logging.getLogger(__name__)


# ==================== Supervisor 内置工具参数定义 ====================

class HandoffParams(BaseModel):
    """handoff_to_agent 工具的参数"""
    agent_name: str = Field(description="要委派任务的 Agent 名称")
    task: str = Field(description="要交给该 Agent 的具体任务描述")


class FinalAnswerParams(BaseModel):
    """final_answer 工具的参数"""
    answer: str = Field(description="最终答案，将直接返回给用户")


# ==================== Supervisor 内置工具实现 ====================

class _HandoffTool(Tool):
    """Supervisor 的任务委派工具（由编排器内部管理）"""

    def __init__(self, orchestrator: "SupervisorOrchestrator"):
        super().__init__(
            name="handoff_to_agent",
            description=(
                "将一个子任务委派给指定的 Worker Agent 执行。"
                "你必须从可用 Worker 列表中选择。"
            ),
            parameters=HandoffParams,
        )
        self._orchestrator = orchestrator

    def run(self, parameters: dict) -> str:
        agent_name = parameters.get("agent_name", "")
        task = parameters.get("task", "")

        logger.info(
            "🤝 Supervisor 委派任务: '%s' → Agent '%s'",
            (task or "")[:60], agent_name,
        )

        try:
            result_msg = self._orchestrator._invoke_worker(agent_name, task)
            return result_msg.content
        except Exception as e:
            error_msg = f"委派给 Agent '{agent_name}' 失败: {e}"
            logger.error("❌ %s", error_msg)
            return error_msg


class _FinalAnswerTool(Tool):
    """Supervisor 的最终答案工具"""

    def __init__(self):
        super().__init__(
            name="final_answer",
            description="当你认为任务已经完成，调用此工具提交最终答案给用户。",
            parameters=FinalAnswerParams,
        )
        self._answer: Optional[str] = None

    def run(self, parameters: dict) -> str:
        self._answer = parameters.get("answer", "")
        logger.info("🎯 Supervisor 提交最终答案 (长度=%d)", len(self._answer))
        return "最终答案已提交。"


# ==================== SupervisorOrchestrator ====================

class SupervisorOrchestrator(BaseOrchestrator):
    """
    Supervisor 模式编排器
    
    一个 Supervisor Agent 动态决策调用哪个 Worker Agent。
    Supervisor 通过工具调用 (handoff_to_agent / final_answer) 来控制流程。
    
    特点：
    - 动态路由：Supervisor 根据任务内容和 Worker 能力自主选择
    - 多轮交互：Supervisor 可以连续调用多个 Worker，汇总结果
    - 超时保护：max_rounds 限制最大轮次，防止无限循环
    
    Attributes:
        supervisor_llm: Supervisor 使用的 LLM
        max_rounds: 最大调度轮次
    """

    def __init__(
        self,
        name: str,
        supervisor_llm: "EasyLLM",
        max_rounds: int = 10,
        temperature: float = 0.7,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        super().__init__(name=name, callback_manager=callback_manager)
        self._supervisor_llm = supervisor_llm
        self._max_rounds = max_rounds
        self._temperature = temperature
        self._context: Optional[SharedContext] = None

        logger.info(
            "👔 SupervisorOrchestrator '%s' 初始化: max_rounds=%d",
            name, max_rounds,
        )

    def _invoke_worker(self, agent_name: str, task: str) -> "AgentMessage":
        """
        被 _HandoffTool 回调调用——安全地执行一个 Worker Agent
        """
        if self._context is None:
            raise OrchestrationError("SharedContext 未初始化")

        # 触发 handoff 回调
        self._trigger_handoff("supervisor", agent_name, task)

        # 给 Worker 注入上下文
        worker_prompt = self._build_worker_prompt(agent_name, task)

        return self._invoke_agent(
            agent_name=agent_name,
            query=worker_prompt,
            context=self._context,
            temperature=self._temperature,
        )

    def run(self, query: str, **kwargs) -> str:
        """
        启动 Supervisor 编排
        
        Args:
            query: 用户原始输入
            
        Returns:
            Supervisor 的最终答案
            
        Raises:
            MaxRoundsExceededError: 达到最大轮次
            OrchestrationError: 编排失败
        """
        self._validate_agents()
        self._context = SharedContext(original_query=query)

        total_start = time.time()

        logger.info(
            "🚀 Supervisor 编排器 '%s' 开始: query='%s', workers=%s",
            self.name, query[:50], self.agent_names,
        )

        # 构建 Supervisor 的工具集
        handoff_tool = _HandoffTool(self)
        final_answer_tool = _FinalAnswerTool()

        supervisor_registry = ToolRegistry()
        supervisor_registry.register_tool(handoff_tool)
        supervisor_registry.register_tool(final_answer_tool)

        # 构建 Supervisor 系统提示
        system_prompt = self._build_supervisor_prompt(query)

        # 延迟导入避免循环引用
        from agent.BasicAgent import BasicAgent

        supervisor_agent = BasicAgent(
            name=f"{self.name}_supervisor",
            llm=self._supervisor_llm,
            system_prompt=system_prompt,
        ).with_tool(supervisor_registry)

        # 多轮调度循环
        current_query = query
        for round_idx in range(1, self._max_rounds + 1):
            logger.info(
                "--- Supervisor 轮次 %d/%d ---",
                round_idx, self._max_rounds,
            )

            try:
                supervisor_agent.invoke(
                    query=current_query,
                    temperature=self._temperature,
                )
            except Exception as e:
                logger.error("❌ Supervisor 调用失败: %s", e)
                raise OrchestrationError(f"Supervisor 调用失败: {e}") from e

            # 检查是否提交了最终答案
            if final_answer_tool._answer is not None:
                result = final_answer_tool._answer
                total_duration_ms = int((time.time() - total_start) * 1000)

                logger.info(
                    "🏁 Supervisor 编排器 '%s' 完成: 轮次=%d, 耗时=%dms, 输出长度=%d",
                    self.name, round_idx, total_duration_ms, len(result),
                )

                self._trigger_orchestrator_end(result, round_idx)
                return result

            # 没有提交最终答案——用上一轮的执行记录作为下一轮输入
            last_result = self._context.get_last_result()
            if last_result:
                current_query = (
                    f"上一轮 Worker 的执行结果:\n{last_result.content}\n\n"
                    f"请继续处理用户的原始需求，或调用 final_answer 提交最终答案。"
                )
            else:
                current_query = "请继续处理任务或调用 final_answer 提交最终答案。"

        # 达到最大轮次
        total_duration_ms = int((time.time() - total_start) * 1000)
        logger.warning(
            "⚠️  Supervisor 编排器 '%s' 达到最大轮次 %d (耗时=%dms)",
            self.name, self._max_rounds, total_duration_ms,
        )

        # 返回最后一个结果作为兜底
        last_result = self._context.get_last_result()
        fallback = last_result.content if last_result else "编排超时，未获得最终答案。"

        self._trigger_orchestrator_end(fallback, self._max_rounds)
        return fallback

    # ==================== Prompt 构建 ====================

    def _build_supervisor_prompt(self, query: str) -> str:
        """构建 Supervisor 的系统提示"""
        worker_descriptions = []
        for name, agent in self._agents.items():
            desc = agent.description or "(无描述)"
            worker_descriptions.append(f"  - {name}: {desc}")
        workers_text = "\n".join(worker_descriptions)

        return (
            f"你是一个任务调度管理器 (Supervisor)。\n"
            f"\n"
            f"你的职责是分析用户的需求，将子任务委派给合适的 Worker Agent。\n"
            f"你不需要自己完成任务，而是调度以下 Worker：\n"
            f"\n"
            f"【可用 Worker 列表】\n"
            f"{workers_text}\n"
            f"\n"
            f"【工作流程】\n"
            f"1. 分析用户需求，拆解为子任务\n"
            f"2. 调用 handoff_to_agent 工具将子任务交给合适的 Worker\n"
            f"3. 查看 Worker 返回结果\n"
            f"4. 如需更多步骤，继续调用 handoff_to_agent\n"
            f"5. 当所有子任务完成后，调用 final_answer 提交最终答案\n"
            f"\n"
            f"【注意事项】\n"
            f"- 只能委派给上述列表中的 Worker\n"
            f"- 给 Worker 的任务描述要清晰具体\n"
            f"- 最终必须调用 final_answer 提交答案\n"
        )

    def _build_worker_prompt(self, agent_name: str, task: str) -> str:
        """构建传递给 Worker 的 prompt"""
        return (
            f"你收到了来自 Supervisor 的任务委派。\n"
            f"\n"
            f"【用户原始需求】\n{self._context.original_query}\n"  # type: ignore
            f"\n"
            f"【分配给你的具体任务】\n{task}\n"
            f"\n"
            f"请专注完成你的任务，输出清晰、详细的结果。"
        )
