
class AgentError(Exception):
    """智能体基础异常类"""
    pass


class ToolRegistryError(AgentError):
    """工具注册表相关异常"""
    pass


class ToolExecutionError(AgentError):
    """工具执行异常"""
    pass


class LLMInvokeError(AgentError):
    """LLM 调用异常"""
    pass


class ParameterValidationError(AgentError):
    """参数验证异常"""
    pass


class MemoryError(AgentError):
    """记忆系统异常"""
    pass


class OutputParseError(AgentError):
    """输出解析异常"""
    pass


class PromptTemplateError(AgentError):
    """提示词模板异常"""
    pass


class RetrieverError(AgentError):
    """检索器异常"""
    pass


class PlanningError(AgentError):
    """规划异常"""
    pass