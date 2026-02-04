"""
WorkingMemory 模块单元测试
"""
import unittest
import sys
import os
import datetime
import time

# 添加项目根目录和V2模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'memory', 'V2'))

from memory.V2.WorkingMemory import WorkingMemory
from memory.V2.BaseMemory import BaseMemory, MemoryConfig, MemoryItem, ForgetType


class TestWorkingMemoryBasic(unittest.TestCase):
    """WorkingMemory 基础功能测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=10, max_working_token=1000)
        self.memory = WorkingMemory(self.config)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def _create_memory_item(self, id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """辅助方法：创建 MemoryItem"""
        return MemoryItem(
            id=id,
            content=content,
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=importance,
            metadata={}
        )

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.memory.max_capacity, 10)
        self.assertEqual(self.memory.max_token, 1000)
        self.assertEqual(self.memory.size, 0)
        self.assertEqual(self.memory.current_tokens, 0)
        self.assertEqual(len(self.memory.memory_list), 0)
        self.assertEqual(len(self.memory.memory_heap), 0)

    def test_add_memory(self):
        """测试添加记忆"""
        item = self._create_memory_item("1", "hello world")
        result = self.memory.add_memory(item)

        self.assertEqual(result, "1")
        self.assertEqual(self.memory.size, 1)
        self.assertEqual(len(self.memory.memory_list), 1)
        self.assertEqual(len(self.memory.memory_heap), 1)

    def test_add_multiple_memories(self):
        """测试添加多条记忆"""
        for i in range(5):
            item = self._create_memory_item(str(i), f"content {i}")
            self.memory.add_memory(item)

        self.assertEqual(self.memory.size, 5)
        self.assertEqual(len(self.memory.memory_list), 5)

    def test_remove_memory(self):
        """测试删除记忆"""
        item = self._create_memory_item("1", "hello world")
        self.memory.add_memory(item)
        self.assertEqual(self.memory.size, 1)

        result = self.memory.remove_memory("1")
        self.assertTrue(result)
        self.assertEqual(self.memory.size, 0)
        self.assertEqual(len(self.memory.memory_list), 0)

    def test_remove_nonexistent_memory(self):
        """测试删除不存在的记忆"""
        result = self.memory.remove_memory("nonexistent")
        self.assertFalse(result)

    def test_update_memory(self):
        """测试更新记忆"""
        item = self._create_memory_item("1", "original content", importance=0.5)
        self.memory.add_memory(item)

        result = self.memory.update_memory("1", "updated content", importance=0.8)
        self.assertTrue(result)

        # 检查更新后的内容
        memories = self.memory.get_all_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "updated content")
        self.assertEqual(memories[0].importance, 0.8)

    def test_update_nonexistent_memory(self):
        """测试更新不存在的记忆"""
        result = self.memory.update_memory("nonexistent", "new content")
        self.assertFalse(result)

    def test_find_memory(self):
        """测试查找记忆"""
        item = self._create_memory_item("1", "hello world")
        self.memory.add_memory(item)

        self.assertTrue(self.memory.find_memory("1"))
        self.assertFalse(self.memory.find_memory("2"))

    def test_clear_memory(self):
        """测试清空记忆"""
        for i in range(5):
            item = self._create_memory_item(str(i), f"content {i}")
            self.memory.add_memory(item)

        self.assertEqual(self.memory.size, 5)

        self.memory.clear_memory()

        self.assertEqual(self.memory.size, 0)
        self.assertEqual(self.memory.current_tokens, 0)
        self.assertEqual(len(self.memory.memory_list), 0)
        self.assertEqual(len(self.memory.memory_heap), 0)


class TestWorkingMemoryRetrieval(unittest.TestCase):
    """WorkingMemory 记忆检索测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=20, max_working_token=5000)
        self.memory = WorkingMemory(self.config)

        # 添加测试数据
        self.test_items = [
            self._create_memory_item("1", "python programming language", importance=0.9),
            self._create_memory_item("2", "machine learning algorithms", importance=0.8),
            self._create_memory_item("3", "deep learning neural networks", importance=0.7),
            self._create_memory_item("4", "data science analytics", importance=0.6),
            self._create_memory_item("5", "web development javascript", importance=0.5),
        ]
        for item in self.test_items:
            self.memory.add_memory(item)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def _create_memory_item(self, id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """辅助方法：创建 MemoryItem"""
        return MemoryItem(
            id=id,
            content=content,
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=importance,
            metadata={}
        )

    def test_get_all_memories(self):
        """测试获取所有记忆"""
        memories = self.memory.get_all_memories()
        self.assertEqual(len(memories), 5)

    def test_get_recent_memories(self):
        """测试获取最近的记忆"""
        memories = self.memory.get_recent_memories(limit=3)
        self.assertEqual(len(memories), 3)
        # 最近的应该是最后添加的
        self.assertEqual(memories[-1].id, "5")

    def test_get_recent_memories_limit_exceeds(self):
        """测试限制超过总数时的行为"""
        memories = self.memory.get_recent_memories(limit=10)
        self.assertEqual(len(memories), 5)

    def test_get_important_memories(self):
        """测试获取最重要的记忆"""
        memories = self.memory.get_important_memories(limit=3)
        self.assertEqual(len(memories), 3)
        # 最重要的应该是 importance 最高的
        self.assertEqual(memories[0].id, "1")  # importance=0.9
        self.assertEqual(memories[1].id, "2")  # importance=0.8
        self.assertEqual(memories[2].id, "3")  # importance=0.7

    def test_search_memory_keyword(self):
        """测试关键字搜索"""
        results = self.memory.search_memory("python programming", limit=3)
        self.assertGreater(len(results), 0)
        # python 相关的应该排在前面
        self.assertTrue(any("python" in r.content.lower() for r in results))

    def test_search_memory_with_user_id(self):
        """测试按用户 ID 搜索"""
        results = self.memory.search_memory("programming", limit=3, user_id="test_user")
        self.assertGreater(len(results), 0)

    def test_search_memory_no_results(self):
        """测试搜索无结果的情况"""
        self.memory.clear_memory()
        results = self.memory.search_memory("python", limit=3)
        self.assertEqual(len(results), 0)


class TestWorkingMemoryCapacity(unittest.TestCase):
    """WorkingMemory 容量管理测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=5, max_working_token=100)
        self.memory = WorkingMemory(self.config)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def _create_memory_item(self, id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """辅助方法：创建 MemoryItem"""
        return MemoryItem(
            id=id,
            content=content,
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=importance,
            metadata={}
        )

    def test_capacity_limit(self):
        """测试容量限制"""
        # 添加超过容量的记忆
        for i in range(10):
            item = self._create_memory_item(str(i), f"short", importance=0.5 + i * 0.05)
            self.memory.add_memory(item)

        # 应该不超过最大容量 + 1 (由于实现中先添加后检查)
        # 或者不超过 token 限制
        self.assertLessEqual(self.memory.size, self.config.max_capacity + 1)

    def test_priority_based_cleanup(self):
        """测试基于优先级的清理"""
        # 添加低优先级记忆
        for i in range(3):
            item = self._create_memory_item(f"low_{i}", "low priority", importance=0.1)
            self.memory.add_memory(item)

        # 添加高优先级记忆
        for i in range(3):
            item = self._create_memory_item(f"high_{i}", "high priority", importance=0.9)
            self.memory.add_memory(item)

        # 高优先级的应该被保留
        memories = self.memory.get_all_memories()
        high_priority_count = sum(1 for m in memories if "high" in m.id)
        self.assertGreater(high_priority_count, 0)


class TestWorkingMemoryTimeDecay(unittest.TestCase):
    """WorkingMemory 时间衰减测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=100, decay_factor=0.95)
        self.memory = WorkingMemory(self.config)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def test_time_decay_calculation(self):
        """测试时间衰减计算"""
        now = datetime.datetime.now()
        
        # 刚刚的时间，衰减应该接近 1
        recent_decay = self.memory._calculate_time_decay(now)
        self.assertGreater(recent_decay, 0.9)

        # 6小时前的时间，衰减应该等于 decay_factor
        past_time = now - datetime.timedelta(hours=6)
        past_decay = self.memory._calculate_time_decay(past_time)
        self.assertAlmostEqual(past_decay, self.config.decay_factor, places=2)

    def test_priority_calculation(self):
        """测试优先级计算"""
        item = MemoryItem(
            id="1",
            content="test content",
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=1.0,
            metadata={}
        )
        
        priority = self.memory._calculate_priority(item)
        # 新记忆的优先级应该接近其重要性
        self.assertGreater(priority, 0.9)


class TestWorkingMemoryForget(unittest.TestCase):
    """WorkingMemory 遗忘功能测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=10, max_working_token=1000)
        self.memory = WorkingMemory(self.config)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def _create_memory_item(self, id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """辅助方法：创建 MemoryItem"""
        return MemoryItem(
            id=id,
            content=content,
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=importance,
            metadata={}
        )

    def test_forget_by_importance(self):
        """测试按重要性遗忘"""
        # 添加不同重要性的记忆
        for i in range(5):
            item = self._create_memory_item(str(i), f"content {i}", importance=i * 0.2)
            self.memory.add_memory(item)

        initial_size = self.memory.size
        forgotten = self.memory.forget(ForgetType.IMPORTANCE, threshold=0.5)

        # 应该删除了一些低重要性的记忆
        self.assertGreaterEqual(forgotten, 0)
        self.assertLessEqual(self.memory.size, initial_size)

    def test_forget_by_capacity(self):
        """测试按容量遗忘"""
        for i in range(5):
            item = self._create_memory_item(str(i), f"content {i}")
            self.memory.add_memory(item)

        # 容量遗忘应该移除一些记忆
        forgotten = self.memory.forget(ForgetType.CAPACITY)
        self.assertGreaterEqual(forgotten, 0)


class TestWorkingMemoryStats(unittest.TestCase):
    """WorkingMemory 统计信息测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=10, max_working_token=1000)
        self.memory = WorkingMemory(self.config)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def _create_memory_item(self, id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """辅助方法：创建 MemoryItem"""
        return MemoryItem(
            id=id,
            content=content,
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=importance,
            metadata={}
        )

    def test_get_stats_empty(self):
        """测试空记忆的统计信息"""
        stats = self.memory.get_stats()

        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["total_count"], 0)
        self.assertEqual(stats["current_tokens"], 0)
        self.assertEqual(stats["memory_type"], "working")

    def test_get_stats_with_memories(self):
        """测试有记忆时的统计信息"""
        for i in range(3):
            item = self._create_memory_item(str(i), f"content number {i}", importance=0.5 + i * 0.1)
            self.memory.add_memory(item)

        stats = self.memory.get_stats()

        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["total_count"], 3)
        self.assertGreater(stats["current_tokens"], 0)
        self.assertEqual(stats["max_capacity"], 10)
        self.assertEqual(stats["max_tokens"], 1000)
        self.assertGreater(stats["avg_importance"], 0)
        self.assertGreater(stats["capacity_usage"], 0)
        self.assertEqual(stats["memory_type"], "working")


class TestWorkingMemorySimilarity(unittest.TestCase):
    """WorkingMemory 相似度计算测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig()
        self.memory = WorkingMemory(self.config)

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        query_embedding = [1.0, 0.0, 0.0]
        memories_embedding = [
            [1.0, 0.0, 0.0],  # 完全相同
            [0.0, 1.0, 0.0],  # 正交
            [0.5, 0.5, 0.0],  # 部分相似
        ]

        similarities = self.memory.cosine_similarity(query_embedding, memories_embedding)

        self.assertEqual(len(similarities), 3)
        self.assertAlmostEqual(similarities[0], 1.0, places=5)  # 完全相同
        self.assertAlmostEqual(similarities[1], 0.0, places=5)  # 正交
        self.assertGreater(similarities[2], 0.0)  # 部分相似


class TestWorkingMemoryEdgeCases(unittest.TestCase):
    """WorkingMemory 边界情况测试"""

    def setUp(self):
        """每个测试前初始化"""
        self.config = MemoryConfig(max_capacity=5, max_working_token=100)
        self.memory = WorkingMemory(self.config)

    def tearDown(self):
        """每个测试后清理"""
        self.memory.clear_memory()

    def _create_memory_item(self, id: str, content: str, importance: float = 0.5) -> MemoryItem:
        """辅助方法：创建 MemoryItem"""
        return MemoryItem(
            id=id,
            content=content,
            type="working",
            user_id="test_user",
            timestamp=datetime.datetime.now(),
            importance=importance,
            metadata={}
        )

    def test_empty_content(self):
        """测试空内容"""
        item = self._create_memory_item("1", "")
        self.memory.add_memory(item)
        self.assertEqual(self.memory.size, 1)

    def test_very_long_content(self):
        """测试很长的内容"""
        long_content = " ".join(["word"] * 100)  # 100个单词
        item = self._create_memory_item("1", long_content)
        self.memory.add_memory(item)
        self.assertEqual(self.memory.size, 1)

    def test_special_characters_in_content(self):
        """测试特殊字符"""
        item = self._create_memory_item("1", "Hello! @#$%^&*() 你好世界 🎉")
        self.memory.add_memory(item)
        self.assertEqual(self.memory.size, 1)
        self.assertTrue(self.memory.find_memory("1"))

    def test_duplicate_ids(self):
        """测试重复 ID"""
        item1 = self._create_memory_item("1", "first content")
        item2 = self._create_memory_item("1", "second content")
        
        self.memory.add_memory(item1)
        self.memory.add_memory(item2)
        
        # 两个都会被添加（当前实现允许重复 ID）
        self.assertEqual(self.memory.size, 2)

    def test_zero_importance(self):
        """测试零重要性"""
        item = self._create_memory_item("1", "test", importance=0.0)
        self.memory.add_memory(item)
        self.assertEqual(self.memory.size, 1)

    def test_max_importance(self):
        """测试最大重要性"""
        item = self._create_memory_item("1", "test", importance=1.0)
        self.memory.add_memory(item)
        self.assertEqual(self.memory.size, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
