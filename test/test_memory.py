"""
Memory 模块单元测试
"""
import unittest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core
core.enable_logging("DEBUG")
from memory.buffer import ConversationBufferMemory
from memory.base import BaseMemory
from core.Message import UserMessage, AssistantMessage


class TestConversationBufferMemory(unittest.TestCase):
    """ConversationBufferMemory 测试"""
    
    def setUp(self):
        """每个测试前初始化"""
        self.memory = ConversationBufferMemory(max_messages=5)
    
    def test_add_message(self):
        """测试添加消息"""
        self.memory.add_user_message("你好")
        self.memory.add_assistant_message("你好！有什么可以帮助你的？")
        
        messages = self.memory.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")
    
    def test_max_messages_limit(self):
        """测试消息数量限制"""
        for i in range(10):
            self.memory.add_user_message(f"消息 {i}")
        
        messages = self.memory.get_messages()
        self.assertEqual(len(messages), 5)
        # 最早的消息应该被移除
        self.assertEqual(messages[0].content, "消息 5")
    
    def test_get_context(self):
        """测试获取上下文"""
        self.memory.add_user_message("我叫张三")
        self.memory.add_assistant_message("你好，张三！")
        
        context = self.memory.get_context()
        self.assertIn("用户", context)
        self.assertIn("助手", context)
        self.assertIn("张三", context)
    
    def test_clear(self):
        """测试清空记忆"""
        self.memory.add_user_message("测试消息")
        self.memory.clear()
        
        self.assertEqual(len(self.memory.get_messages()), 0)
    
    def test_custom_prefixes(self):
        """测试自定义前缀"""
        memory = ConversationBufferMemory(
            human_prefix="Human",
            ai_prefix="AI"
        )
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi!")
        
        context = memory.get_context()
        self.assertIn("Human:", context)
        self.assertIn("AI:", context)
    
    def test_get_last_n_messages(self):
        """测试获取最后 N 条消息"""
        for i in range(10):
            self.memory.add_user_message(f"消息 {i}")
        
        last_3 = self.memory.get_last_n_messages(3)
        self.assertEqual(len(last_3), 3)
    
    def test_memory_variables(self):
        """测试记忆变量"""
        self.memory.add_user_message("测试")
        variables = self.memory.get_memory_variables()
        
        self.assertIn("history", variables)
        self.assertIsInstance(variables["history"], str)


class TestConversationSummaryMemory(unittest.TestCase):
    """ConversationSummaryMemory 测试（需要 LLM，这里只测试基本功能）"""
    
    def test_import(self):
        """测试导入"""
        from memory.summary import ConversationSummaryMemory
        self.assertTrue(True)


class TestBaseMemory(unittest.TestCase):
    """BaseMemory 抽象类测试"""
    
    def test_is_abstract(self):
        """测试 BaseMemory 是抽象类"""
        with self.assertRaises(TypeError):
            BaseMemory()


if __name__ == "__main__":
    unittest.main(verbosity=2)
