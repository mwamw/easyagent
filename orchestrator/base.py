"""
多 Agent 协作 — 编排器基类

BaseOrchestrator 提供所有编排模式共享的能力：
- Agent 注册与管理
- 安全调用（统一异常处理 + 日志 + 回调）
- 共享上下文管理
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

from .message import AgentMessage
from .context import SharedContext
from .exceptions import AgentNotFoundError, OrchestrationError

if TYPE_CHECKING:
    from core.agent import BaseAgent
    from core.callbacks import CallbackManager

logger = logging.getLogger(__name__)


class BaseOrchestrator(ABC):
    """
    编排器抽象基类
    
    所有编排模式（Sequential、Supervisor、GroupChat）均继承此类。
    基类负责通用的 Agent 管理、安全调用、日志和回调。
    
    Attributes:
        name: 编排器名称（用于日志标识）
        agents: 已注册的 Agent 字典 {name: agent}
        shared_context: 共享上下文（协作黑板）
        callback_manager: 回调管理器
    """

    def __init__(
        self,
        name: str,
        callback_manager: Optional["CallbackManager"] = None,
    ):
        self.name = name
        self._agents: dict[str, "BaseAgent"] = {}
        self._callback_manager = callback_manager
        
        logger.info("🎯 编排器 '%s' (%s) 初始化完成", name, self.__class__.__name__)

    # ==================== Agent 管理 ====================

    def add_agent(self, name: str, agent: "BaseAgent") -> "BaseOrchestrator":
        """
        注册一个 Agent 到编排器
        
        Args:
            name: Agent 在编排器中的角色名（可与 agent.name 不同）
            agent: BaseAgent 实例
            
        Returns:
            self（支持链式调用）
            
        Raises:
            OrchestrationError: 重复注册
        """
        if name in self._agents:
            raise OrchestrationError(f"Agent '{name}' 已存在于编排器 '{self.name}' 中")
        
        self._agents[name] = agent
        logger.info(
            "📎 编排器 '%s': 注册 Agent '%s' (类型=%s, 描述=%s)",
            self.name, name, type(agent).__name__,
            agent.description or "(无)"
        )
        return self

    def remove_agent(self, name: str) -> None:
        """移除一个 Agent"""
        if name not in self._agents:
            raise AgentNotFoundError(f"编排器 '{self.name}' 中未找到 Agent '{name}'")
        del self._agents[name]
        logger.info("📎 编排器 '%s': 移除 Agent '%s'", self.name, name)

    def get_agent(self, name: str) -> "BaseAgent":
        """获取指定名称的 Agent"""
        if name not in self._agents:
            available = list(self._agents.keys())
            raise AgentNotFoundError(
                f"编排器 '{self.name}' 中未找到 Agent '{name}'。可用: {available}"
            )
        return self._agents[name]

    @property
    def agents(self) -> dict[str, "BaseAgent"]:
        """返回已注册 Agent 字典（只读副本）"""
        return dict(self._agents)

    @property
    def agent_names(self) -> list[str]:
        """返回已注册 Agent 名称列表"""
        return list(self._agents.keys())

    # ==================== 安全调用 ====================

    def _invoke_agent(
        self,
        agent_name: str,
        query: str,
        context: SharedContext,
        temperature: float = 0.7,
        max_iter: int = 10,
    ) -> AgentMessage:
        """
        安全调用一个 Agent 并记录到共享上下文
        
        统一处理：
        1. Agent 查找
        2. 调用前日志
        3. 调用 + 计时
        4. 调用后日志
        5. 异常封装
        6. 回调触发
        7. 结果写入 SharedContext
        
        Args:
            agent_name: Agent 角色名
            query: 发送给 Agent 的输入
            context: 共享上下文
            temperature: 温度参数
            max_iter: 最大迭代次数
            
        Returns:
            AgentMessage (msg_type="result")
            
        Raises:
            OrchestrationError: Agent 调用失败
        """
        agent = self.get_agent(agent_name)

        # 记录 task 消息到上下文
        context.add(
            sender="orchestrator",
            receiver=agent_name,
            content=query,
            msg_type="task",
        )

        logger.info(
            "▶️  编排器 '%s': 调用 Agent '%s' (query 长度=%d)",
            self.name, agent_name, len(query)
        )

        start_time = time.time()
        try:
            result = agent.invoke(
                query=query,
                max_iter=max_iter,
                temperature=temperature,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "✅ 编排器 '%s': Agent '%s' 完成 (耗时=%dms, 输出长度=%d)",
                self.name, agent_name, duration_ms, len(result)
            )

            # 记录结果到上下文
            msg = context.add(
                sender=agent_name,
                receiver="orchestrator",
                content=result,
                msg_type="result",
                duration_ms=duration_ms,
            )

            # 回调
            if self._callback_manager:
                self._callback_manager.on_agent_end(
                    agent_name, result, success=True
                )

            return msg

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "❌ 编排器 '%s': Agent '%s' 调用失败 (耗时=%dms): %s",
                self.name, agent_name, duration_ms, e
            )

            # 错误也记录到上下文
            context.add(
                sender=agent_name,
                receiver="orchestrator",
                content=f"执行失败: {e}",
                msg_type="result",
                duration_ms=duration_ms,
                error=str(e),
            )

            if self._callback_manager:
                self._callback_manager.on_agent_end(
                    agent_name, "", success=False, error=e
                )

            raise OrchestrationError(
                f"Agent '{agent_name}' 调用失败: {e}"
            ) from e

    # ==================== 回调辅助 ====================

    def _trigger_handoff(self, from_agent: str, to_agent: str, task: str) -> None:
        """触发 handoff 回调"""
        if self._callback_manager:
            self._callback_manager.on_handoff(from_agent, to_agent, task)

    def _trigger_orchestrator_end(self, result: str, rounds: int) -> None:
        """触发编排器结束回调"""
        if self._callback_manager:
            self._callback_manager.on_orchestrator_end(self.name, result, rounds)

    # ==================== 校验 ====================

    def _validate_agents(self, required_names: Optional[list[str]] = None) -> None:
        """校验 Agent 配置"""
        if not self._agents:
            raise OrchestrationError(f"编排器 '{self.name}' 中没有注册任何 Agent")

        if required_names:
            missing = [n for n in required_names if n not in self._agents]
            if missing:
                raise AgentNotFoundError(
                    f"编排器 '{self.name}' 缺少必要 Agent: {missing}"
                )

    # ==================== 抽象接口 ====================

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """
        执行编排
        
        Args:
            query: 用户原始输入
            **kwargs: 编排模式特有参数
            
        Returns:
            最终输出文本
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"agents={self.agent_names})"
        )
