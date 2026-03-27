"""
多 Agent 协作 — 异常定义
"""
from core.Exception import AgentError


class OrchestrationError(AgentError):
    """编排器通用异常"""
    pass


class AgentNotFoundError(OrchestrationError):
    """在编排器中未找到指定 Agent"""
    pass


class MaxRoundsExceededError(OrchestrationError):
    """达到最大轮次仍未完成"""
    pass


class HandoffError(OrchestrationError):
    """Agent 间任务交接失败"""
    pass
