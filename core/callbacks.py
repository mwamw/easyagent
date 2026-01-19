"""
回调系统模块

提供 Agent 执行过程中的钩子回调。
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CallbackEvent:
    """回调事件数据"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


class BaseCallback(ABC):
    """
    回调基类
    
    定义 Agent 执行过程中的回调接口。
    子类可以选择性覆盖需要的回调方法。
    """
    
    def on_agent_start(self, agent_name: str, query: str, **kwargs) -> None:
        """
        Agent 开始执行时调用
        
        Args:
            agent_name: Agent 名称
            query: 用户输入
        """
        pass
    
    def on_agent_end(
        self, 
        agent_name: str, 
        output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        """
        Agent 执行结束时调用
        
        Args:
            agent_name: Agent 名称
            output: 输出结果
            success: 是否成功
            error: 如果失败，错误信息
        """
        pass
    
    def on_llm_start(self, messages: List[Dict], **kwargs) -> None:
        """
        LLM 调用开始时
        
        Args:
            messages: 发送给 LLM 的消息列表
        """
        pass
    
    def on_llm_end(self, response: str, **kwargs) -> None:
        """
        LLM 调用结束时
        
        Args:
            response: LLM 返回的响应
        """
        pass
    
    def on_tool_start(self, tool_name: str, tool_input: Dict, **kwargs) -> None:
        """
        工具调用开始时
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数
        """
        pass
    
    def on_tool_end(
        self, 
        tool_name: str, 
        tool_output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        """
        工具调用结束时
        
        Args:
            tool_name: 工具名称
            tool_output: 工具输出
            success: 是否成功
            error: 如果失败，错误信息
        """
        pass
    
    def on_chain_start(self, chain_type: str, inputs: Dict, **kwargs) -> None:
        """思考链/推理链开始时"""
        pass
    
    def on_chain_end(self, outputs: Dict, **kwargs) -> None:
        """思考链/推理链结束时"""
        pass
    
    def on_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """
        发生错误时
        
        Args:
            error: 异常对象
            context: 错误上下文描述
        """
        pass


class LoggingCallback(BaseCallback):
    """
    日志回调
    
    将 Agent 执行过程记录到日志。
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        初始化日志回调
        
        Args:
            log_level: 日志级别
        """
        self.log_level = log_level
        self.logger = logging.getLogger("EasyAgent.Callback")
    
    def on_agent_start(self, agent_name: str, query: str, **kwargs) -> None:
        self.logger.log(self.log_level, f"[Agent Start] {agent_name}: {query[:100]}...")
    
    def on_agent_end(
        self, 
        agent_name: str, 
        output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        if success:
            self.logger.log(self.log_level, f"[Agent End] {agent_name}: 成功")
        else:
            self.logger.warning(f"[Agent End] {agent_name}: 失败 - {error}")
    
    def on_tool_start(self, tool_name: str, tool_input: Dict, **kwargs) -> None:
        self.logger.log(self.log_level, f"[Tool Start] {tool_name}: {tool_input}")
    
    def on_tool_end(
        self, 
        tool_name: str, 
        tool_output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        if success:
            output_preview = tool_output[:100] + "..." if len(tool_output) > 100 else tool_output
            self.logger.log(self.log_level, f"[Tool End] {tool_name}: {output_preview}")
        else:
            self.logger.warning(f"[Tool End] {tool_name}: 失败 - {error}")
    
    def on_error(self, error: Exception, context: str = "", **kwargs) -> None:
        self.logger.error(f"[Error] {context}: {error}")


class StreamingCallback(BaseCallback):
    """
    流式输出回调
    
    用于实时输出 Agent 执行过程。
    """
    
    def __init__(self, print_fn=None, verbose: bool = True):
        """
        初始化流式回调
        
        Args:
            print_fn: 自定义打印函数，默认使用 print
            verbose: 是否详细输出
        """
        self.print_fn = print_fn or print
        self.verbose = verbose
    
    def on_agent_start(self, agent_name: str, query: str, **kwargs) -> None:
        self.print_fn(f"\n🤖 {agent_name} 开始处理...")
        if self.verbose:
            self.print_fn(f"   输入: {query[:200]}{'...' if len(query) > 200 else ''}")
    
    def on_agent_end(
        self, 
        agent_name: str, 
        output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        if success:
            self.print_fn(f"✅ {agent_name} 完成")
        else:
            self.print_fn(f"❌ {agent_name} 失败: {error}")
    
    def on_tool_start(self, tool_name: str, tool_input: Dict, **kwargs) -> None:
        self.print_fn(f"   🔧 调用工具: {tool_name}")
        if self.verbose:
            self.print_fn(f"      参数: {tool_input}")
    
    def on_tool_end(
        self, 
        tool_name: str, 
        tool_output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        if success:
            preview = tool_output[:150] + "..." if len(tool_output) > 150 else tool_output
            self.print_fn(f"      结果: {preview}")
        else:
            self.print_fn(f"      ❌ 失败: {error}")


class MetricsCallback(BaseCallback):
    """
    指标收集回调
    
    收集 Agent 执行过程中的各种指标。
    """
    
    def __init__(self):
        """初始化指标回调"""
        self.metrics: Dict[str, Any] = {
            "agent_calls": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "errors": 0,
            "agent_durations": [],
            "tool_durations": [],
            "tools_used": {},
        }
        self._agent_start_times: Dict[str, datetime] = {}
        self._tool_start_times: Dict[str, datetime] = {}
    
    def on_agent_start(self, agent_name: str, query: str, **kwargs) -> None:
        self.metrics["agent_calls"] += 1
        self._agent_start_times[agent_name] = datetime.now()
    
    def on_agent_end(
        self, 
        agent_name: str, 
        output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        if agent_name in self._agent_start_times:
            duration = (datetime.now() - self._agent_start_times[agent_name]).total_seconds()
            self.metrics["agent_durations"].append(duration)
            del self._agent_start_times[agent_name]
        
        if not success:
            self.metrics["errors"] += 1
    
    def on_llm_start(self, messages: List[Dict], **kwargs) -> None:
        self.metrics["llm_calls"] += 1
    
    def on_tool_start(self, tool_name: str, tool_input: Dict, **kwargs) -> None:
        self.metrics["tool_calls"] += 1
        self.metrics["tools_used"][tool_name] = self.metrics["tools_used"].get(tool_name, 0) + 1
        self._tool_start_times[tool_name] = datetime.now()
    
    def on_tool_end(
        self, 
        tool_name: str, 
        tool_output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        if tool_name in self._tool_start_times:
            duration = (datetime.now() - self._tool_start_times[tool_name]).total_seconds()
            self.metrics["tool_durations"].append(duration)
            del self._tool_start_times[tool_name]
        
        if not success:
            self.metrics["errors"] += 1
    
    def on_error(self, error: Exception, context: str = "", **kwargs) -> None:
        self.metrics["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取收集的指标"""
        metrics = self.metrics.copy()
        
        # 计算平均值
        if metrics["agent_durations"]:
            metrics["avg_agent_duration"] = sum(metrics["agent_durations"]) / len(metrics["agent_durations"])
        if metrics["tool_durations"]:
            metrics["avg_tool_duration"] = sum(metrics["tool_durations"]) / len(metrics["tool_durations"])
        
        return metrics
    
    def reset(self) -> None:
        """重置指标"""
        self.metrics = {
            "agent_calls": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "errors": 0,
            "agent_durations": [],
            "tool_durations": [],
            "tools_used": {},
        }


class CallbackManager:
    """
    回调管理器
    
    管理多个回调并统一触发。
    """
    
    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        """
        初始化回调管理器
        
        Args:
            callbacks: 回调列表
        """
        self.callbacks: List[BaseCallback] = callbacks or []
    
    def add_callback(self, callback: BaseCallback) -> None:
        """添加回调"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: BaseCallback) -> None:
        """移除回调"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_agent_start(self, agent_name: str, query: str, **kwargs) -> None:
        for cb in self.callbacks:
            try:
                cb.on_agent_start(agent_name, query, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
    
    def on_agent_end(
        self, 
        agent_name: str, 
        output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        for cb in self.callbacks:
            try:
                cb.on_agent_end(agent_name, output, success, error, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
    
    def on_llm_start(self, messages: List[Dict], **kwargs) -> None:
        for cb in self.callbacks:
            try:
                cb.on_llm_start(messages, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
    
    def on_llm_end(self, response: str, **kwargs) -> None:
        for cb in self.callbacks:
            try:
                cb.on_llm_end(response, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
    
    def on_tool_start(self, tool_name: str, tool_input: Dict, **kwargs) -> None:
        for cb in self.callbacks:
            try:
                cb.on_tool_start(tool_name, tool_input, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
    
    def on_tool_end(
        self, 
        tool_name: str, 
        tool_output: str, 
        success: bool = True,
        error: Optional[Exception] = None,
        **kwargs
    ) -> None:
        for cb in self.callbacks:
            try:
                cb.on_tool_end(tool_name, tool_output, success, error, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
    
    def on_error(self, error: Exception, context: str = "", **kwargs) -> None:
        for cb in self.callbacks:
            try:
                cb.on_error(error, context, **kwargs)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")
