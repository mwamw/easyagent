"""
Callbacks 回调系统单元测试
"""
import unittest
import sys
import os
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.callbacks import (
    BaseCallback,
    LoggingCallback,
    StreamingCallback,
    MetricsCallback,
    CallbackManager,
)


class TestLoggingCallback(unittest.TestCase):
    """LoggingCallback 测试"""
    
    def setUp(self):
        self.callback = LoggingCallback(log_level=logging.DEBUG)
    
    def test_on_agent_start(self):
        """测试 agent 开始回调"""
        # 不应该抛出异常
        self.callback.on_agent_start("test_agent", "测试查询")
    
    def test_on_agent_end_success(self):
        """测试 agent 成功结束回调"""
        self.callback.on_agent_end("test_agent", "输出结果", success=True)
    
    def test_on_agent_end_failure(self):
        """测试 agent 失败结束回调"""
        error = Exception("测试错误")
        self.callback.on_agent_end("test_agent", "", success=False, error=error)
    
    def test_on_tool_start(self):
        """测试工具开始回调"""
        self.callback.on_tool_start("calculator", {"expression": "1+1"})
    
    def test_on_tool_end(self):
        """测试工具结束回调"""
        self.callback.on_tool_end("calculator", "2")


class TestStreamingCallback(unittest.TestCase):
    """StreamingCallback 测试"""
    
    def setUp(self):
        self.outputs = []
        self.callback = StreamingCallback(
            print_fn=lambda x: self.outputs.append(x),
            verbose=True
        )
    
    def test_on_agent_start(self):
        """测试 agent 开始输出"""
        self.callback.on_agent_start("my_agent", "你好")
        
        self.assertTrue(any("my_agent" in o for o in self.outputs))
    
    def test_on_tool_start(self):
        """测试工具调用输出"""
        self.callback.on_tool_start("search", {"query": "测试"})
        
        self.assertTrue(any("search" in o for o in self.outputs))


class TestMetricsCallback(unittest.TestCase):
    """MetricsCallback 测试"""
    
    def setUp(self):
        self.callback = MetricsCallback()
    
    def test_agent_call_counting(self):
        """测试 agent 调用计数"""
        self.callback.on_agent_start("agent1", "query1")
        self.callback.on_agent_start("agent2", "query2")
        
        metrics = self.callback.get_metrics()
        self.assertEqual(metrics["agent_calls"], 2)
    
    def test_tool_call_counting(self):
        """测试工具调用计数"""
        self.callback.on_tool_start("tool1", {})
        self.callback.on_tool_end("tool1", "result1")
        self.callback.on_tool_start("tool2", {})
        self.callback.on_tool_end("tool2", "result2")
        
        metrics = self.callback.get_metrics()
        self.assertEqual(metrics["tool_calls"], 2)
    
    def test_tools_used_tracking(self):
        """测试工具使用追踪"""
        self.callback.on_tool_start("calculator", {})
        self.callback.on_tool_start("calculator", {})
        self.callback.on_tool_start("search", {})
        
        metrics = self.callback.get_metrics()
        self.assertEqual(metrics["tools_used"]["calculator"], 2)
        self.assertEqual(metrics["tools_used"]["search"], 1)
    
    def test_error_counting(self):
        """测试错误计数"""
        self.callback.on_error(Exception("error1"))
        self.callback.on_agent_end("agent", "", success=False, error=Exception("error2"))
        
        metrics = self.callback.get_metrics()
        self.assertEqual(metrics["errors"], 2)
    
    def test_reset(self):
        """测试重置指标"""
        self.callback.on_agent_start("agent", "query")
        self.callback.reset()
        
        metrics = self.callback.get_metrics()
        self.assertEqual(metrics["agent_calls"], 0)


class TestCallbackManager(unittest.TestCase):
    """CallbackManager 测试"""
    
    def test_multiple_callbacks(self):
        """测试管理多个回调"""
        metrics = MetricsCallback()
        outputs = []
        streaming = StreamingCallback(print_fn=lambda x: outputs.append(x))
        
        manager = CallbackManager([metrics, streaming])
        manager.on_agent_start("agent", "query")
        
        self.assertEqual(metrics.get_metrics()["agent_calls"], 1)
        self.assertTrue(len(outputs) > 0)
    
    def test_add_remove_callback(self):
        """测试添加和移除回调"""
        manager = CallbackManager()
        callback = MetricsCallback()
        
        manager.add_callback(callback)
        self.assertEqual(len(manager.callbacks), 1)
        
        manager.remove_callback(callback)
        self.assertEqual(len(manager.callbacks), 0)
    
    def test_callback_error_isolation(self):
        """测试回调错误隔离"""
        class BrokenCallback(BaseCallback):
            def on_agent_start(self, agent_name, query, **kwargs):
                raise Exception("Broken!")
        
        metrics = MetricsCallback()
        manager = CallbackManager([BrokenCallback(), metrics])
        
        # 不应该因为一个回调失败而影响其他回调
        manager.on_agent_start("agent", "query")
        self.assertEqual(metrics.get_metrics()["agent_calls"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
