"""
MemoryManage 综合测试程序
覆盖:
  - 初始化 (启用/禁用各类记忆)
  - 添加记忆 (各类型 / 自动分类 / 无效类型)
  - 删除记忆 (remove_memory)
  - 搜索记忆 (单类型/多类型搜索)
  - 更新记忆 (update_memory)
  - 查找记忆 (find_memory)
  - 遗忘策略 (importance/time/capacity)
  - 获取所有记忆 (get_all_memories)
  - 统计信息 (get_memory_stats)
  - 清空记忆 (clear_memories)
  - 合并记忆 (merge_memories)
  - 同步/加载 (sync_memories / load_memories)
  - 分类辅助方法 (_classify_memory_type / _is_episodic / _is_semantic)
  - 边界情况
"""
import sys
import os
import uuid
import traceback
from datetime import datetime, timedelta

# 添加 memory/V2 到路径 (与 MemoryManage.py 内部导入方式一致)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_v2_path = os.path.join(_project_root, 'memory', 'V2')
if _v2_path not in sys.path:
    sys.path.insert(0, _v2_path)
# SemanticMemory → Extractor → agent 依赖
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from MemoryManage import MemoryManage
from BaseMemory import MemoryConfig, MemoryItem, ForgetType
from WorkingMemory import WorkingMemory
from EpisodicMemory import EpisodicMemory
from PerceptualMemory import PerceptualMemory
from Store.SQLiteDocumentStore import SQLiteDocumentStore
from Store.QdrantVectorStore import QdrantVectorStore
from Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
from typing import Optional
import numpy as np


# ==================== 颜色工具 ====================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}")

def print_pass(msg, detail=""):
    print(f"  {Colors.GREEN}✅ PASS{Colors.END} - {msg}" + (f" ({detail})" if detail else ""))

def print_fail(msg, detail=""):
    print(f"  {Colors.RED}❌ FAIL{Colors.END} - {msg}" + (f" ({detail})" if detail else ""))


# ==================== 测试类 ====================
class TestMemoryManage:
    def __init__(self):
        print_section("初始化测试环境")
        self.passed = 0
        self.failed = 0
        self.errors = []

        # 共享配置
        self.config = MemoryConfig(max_capacity=10)
        self.user_id = "test_user"

        # 创建嵌入模型 (共享)
        self.embedding_model = HuggingfaceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 创建 EpisodicMemory 依赖
        self.episodic_doc_store = SQLiteDocumentStore(
            db_path="/home/wxd/LLM/EasyAgent/test/db/test_mm_episodic.db"
        )
        self.episodic_vector_store = QdrantVectorStore(
            way="memory", collection_name="test_mm_episodic", vector_size=384
        )
        self.episodic_memory = EpisodicMemory(
            config=self.config,
            document_store=self.episodic_doc_store,
            vector_store=self.episodic_vector_store,
            embedding_model=self.embedding_model
        )

        # 创建 PerceptualMemory 依赖
        self.perceptual_doc_store = SQLiteDocumentStore(
            db_path="/home/wxd/LLM/EasyAgent/test/db/test_mm_perceptual.db"
        )
        self.perceptual_vector_stores = {
            "text": QdrantVectorStore(way="memory", collection_name="test_mm_perc_text", vector_size=384),
            "image": QdrantVectorStore(way="memory", collection_name="test_mm_perc_image", vector_size=512),
        }
        self.perceptual_memory = PerceptualMemory(
            memory_config=self.config,
            document_store=self.perceptual_doc_store,
            vector_stores=self.perceptual_vector_stores,
            embedding_model=self.embedding_model,
            supported_modalities=["text", "image"]
        )

        # 创建 WorkingMemory (纯内存)
        self.working_memory = WorkingMemory(config=self.config, embedding_model=self.embedding_model)

        # 创建 MemoryManage 实例 (working + episodic, 不启用 semantic)
        self.mm = MemoryManage(
            config=self.config,
            user_id=self.user_id,
            enable_working=True,
            working_memory=self.working_memory,
            enable_episodic=True,
            episodic_memory=self.episodic_memory,
            enable_semantic=False,
            enable_perceptual=True,
            perceptual_memory=self.perceptual_memory,
        )

        print_pass("测试环境初始化完成")

    def _assert(self, condition, msg, detail=""):
        if condition:
            print_pass(msg, detail)
            self.passed += 1
        else:
            print_fail(msg, detail)
            self.failed += 1
            self.errors.append(f"{msg}" + (f" ({detail})" if detail else ""))

    def _reset(self):
        """重置所有记忆到干净状态"""
        self.mm.clear_memories()

    # ==================== 1. 初始化测试 ====================
    def test_init_with_all_types(self):
        print_section("1. 初始化 - 启用 working + episodic + perceptual")

        self._assert("working" in self.mm.memory_types, "working 记忆已启用")
        self._assert("episodic" in self.mm.memory_types, "episodic 记忆已启用")
        self._assert("perceptual" in self.mm.memory_types, "perceptual 记忆已启用")
        self._assert("semantic" not in self.mm.memory_types, "semantic 记忆未启用")
        self._assert(self.mm.user_id == "test_user", f"user_id 正确: {self.mm.user_id}")

    def test_init_only_working(self):
        print_section("2. 初始化 - 仅启用 working")

        mm = MemoryManage(
            config=self.config,
            user_id="only_working_user",
            enable_working=True,
            working_memory=WorkingMemory(self.config),
            enable_episodic=False,
            enable_semantic=False,
            enable_perceptual=False,
        )
        self._assert(len(mm.memory_types) == 1, f"仅启用1种记忆 (实际={len(mm.memory_types)})")
        self._assert("working" in mm.memory_types, "working 记忆已启用")

    def test_init_none_enabled(self):
        print_section("3. 初始化 - 全部禁用")

        mm = MemoryManage(
            config=self.config,
            user_id="no_memory_user",
            enable_working=False,
            enable_episodic=False,
            enable_semantic=False,
            enable_perceptual=False,
        )
        self._assert(len(mm.memory_types) == 0, f"无记忆类型启用 (实际={len(mm.memory_types)})")

    # ==================== 2. 添加记忆测试 ====================
    def test_add_working_memory(self):
        print_section("4. 添加 working 记忆")
        self._reset()

        mid = self.mm.add_memory(
            content="这是一条工作记忆",
            memory_type="working",
            importance=0.6
        )
        self._assert(mid != "", f"添加 working 记忆成功, ID={mid[:8]}...")
        self._assert(self.mm.find_memory(mid), "记忆可被找到")

    def test_add_episodic_memory(self):
        print_section("5. 添加 episodic 记忆")
        self._reset()

        mid = self.mm.add_memory(
            content="今天去公园散步，看到了美丽的花朵。",
            memory_type="episodic",
            importance=0.7,
            metadata={"session_id": "session_001"}
        )
        self._assert(mid != "", f"添加 episodic 记忆成功, ID={mid[:8]}...")
        self._assert(self.mm.find_memory(mid), "episodic 记忆可被找到")

    def test_add_perceptual_memory(self):
        print_section("6. 添加 perceptual 记忆")
        self._reset()

        mid = self.mm.add_memory(
            content="一段关于天气的文本感知",
            memory_type="perceptual",
            importance=0.5,
            metadata={"modality": "text"}
        )
        self._assert(mid != "", f"添加 perceptual 记忆成功, ID={mid[:8]}...")
        self._assert(self.mm.find_memory(mid), "perceptual 记忆可被找到")

    def test_add_invalid_type(self):
        print_section("7. 添加无效类型记忆")
        self._reset()

        try:
            self.mm.add_memory(
                content="无效类型",
                memory_type="nonexistent",
                importance=0.5
            )
            self._assert(False, "应抛出 ValueError")
        except ValueError as e:
            self._assert(True, f"正确抛出 ValueError: {str(e)[:40]}")

    # ==================== 3. 自动分类测试 ====================
    def test_auto_classify_episodic(self):
        print_section("8. 自动分类 - 情景记忆")
        self._reset()

        result = self.mm._classify_memory_type("昨天我去了超市")
        self._assert(result == "episodic", f"'昨天我去了超市' → {result}")

        result2 = self.mm._classify_memory_type("记得那次旅行")
        self._assert(result2 == "episodic", f"'记得那次旅行' → {result2}")

    def test_auto_classify_semantic(self):
        print_section("9. 自动分类 - 语义记忆")

        result = self.mm._classify_memory_type("面向对象编程的定义是什么")
        self._assert(result == "semantic", f"'面向对象编程的定义...' → {result}")

        result2 = self.mm._classify_memory_type("机器学习的基本概念")
        self._assert(result2 == "semantic", f"'机器学习的基本概念' → {result2}")

    def test_auto_classify_working(self):
        print_section("10. 自动分类 - 工作记忆 (默认)")

        result = self.mm._classify_memory_type("你好，世界")
        self._assert(result == "working", f"'你好，世界' → {result}")

    def test_auto_classify_perceptual_by_metadata(self):
        print_section("11. 自动分类 - 感知记忆 (根据metadata)")

        result = self.mm._classify_memory_type("图片数据", metadata={"raw_data": "./image.jpg"})
        self._assert(result == "perceptual", f"'图片数据' + raw_data → {result}")

    def test_auto_classify_by_metadata_type(self):
        print_section("12. 自动分类 - metadata中指定type")

        result = self.mm._classify_memory_type("任意内容", metadata={"type": "episodic"})
        self._assert(result == "episodic", f"metadata.type='episodic' → {result}")

    # ==================== 4. 删除记忆测试 ====================
    def test_remove_working_memory(self):
        print_section("13. 删除 working 记忆")
        self._reset()

        mid = self.mm.add_memory(content="待删除的working记忆", memory_type="working", importance=0.5)
        self._assert(self.mm.find_memory(mid), "删除前记忆存在")

        result = self.mm.remove_memory(mid)
        self._assert(result == True, "remove_memory 返回True")
        self._assert(not self.mm.find_memory(mid), "删除后记忆不存在")

    def test_remove_episodic_memory(self):
        print_section("14. 删除 episodic 记忆")
        self._reset()

        mid = self.mm.add_memory(content="待删除的episodic记忆", memory_type="episodic", importance=0.5)
        self._assert(self.mm.find_memory(mid), "删除前记忆存在")

        result = self.mm.remove_memory(mid)
        self._assert(result == True, "remove_memory 返回True")
        self._assert(not self.mm.find_memory(mid), "删除后记忆不存在")

    def test_remove_nonexistent(self):
        print_section("15. 删除不存在的记忆")
        self._reset()

        result = self.mm.remove_memory("nonexistent_id_xyz")
        self._assert(result == False, "删除不存在的记忆返回False")

    # ==================== 5. 搜索记忆测试 ====================
    def test_search_single_type(self):
        print_section("16. 搜索 - 单类型 (working)")
        self._reset()

        self.mm.add_memory(content="Python是一门编程语言", memory_type="working", importance=0.8)
        self.mm.add_memory(content="Java也是编程语言", memory_type="working", importance=0.6)
        self.mm.add_memory(content="今天天气真好", memory_type="working", importance=0.4)

        results = self.mm.search_memory("编程语言", memory_types=["working"], limit=3)
        self._assert(len(results) > 0, f"搜索到 {len(results)} 条结果")

    def test_search_multi_type(self):
        print_section("17. 搜索 - 多类型 (working + episodic)")
        self._reset()

        self.mm.add_memory(content="工作记忆：学习编程", memory_type="working", importance=0.7)
        self.mm.add_memory(content="情景记忆：今天学了编程", memory_type="episodic", importance=0.6)

        results = self.mm.search_memory("编程", memory_types=["working", "episodic"], limit=5)
        self._assert(len(results) > 0, f"跨类型搜索到 {len(results)} 条结果")

    def test_search_all_types(self):
        print_section("18. 搜索 - 所有类型 (不指定memory_types)")
        self._reset()

        self.mm.add_memory(content="working: 机器学习", memory_type="working", importance=0.8)
        self.mm.add_memory(content="episodic: 今天学了机器学习", memory_type="episodic", importance=0.7)

        results = self.mm.search_memory("机器学习", limit=10)
        self._assert(len(results) > 0, f"全类型搜索到 {len(results)} 条结果")

    def test_search_empty(self):
        print_section("19. 搜索 - 空记忆状态")
        self._reset()

        results = self.mm.search_memory("任意查询", limit=5)
        self._assert(len(results) == 0, "空状态搜索返回空列表")

    # ==================== 6. 更新记忆测试 ====================
    def test_update_working_memory(self):
        print_section("20. 更新 working 记忆")
        self._reset()

        mid = self.mm.add_memory(content="原始内容", memory_type="working", importance=0.5)

        result = self.mm.update_memory(mid, content="更新后的内容", importance=0.9)
        self._assert(result == True, "update_memory 返回True")
        
    def test_update_episodic_memory(self):
        print_section("21. 更新 episodic 记忆")
        self._reset()

        mid = self.mm.add_memory(content="原始情景记忆", memory_type="episodic", importance=0.5)

        result = self.mm.update_memory(mid, content="更新后的情景记忆", importance=0.8)
        self._assert(result == True, "update_memory (episodic) 返回True")

    def test_update_nonexistent(self):
        print_section("22. 更新不存在的记忆")
        self._reset()

        result = self.mm.update_memory("nonexistent_id", content="无效更新")
        self._assert(result == False, "更新不存在的记忆返回False")

    # ==================== 7. 查找记忆测试 ====================
    def test_find_memory_exists(self):
        print_section("23. 查找存在的记忆")
        self._reset()

        mid_w = self.mm.add_memory(content="working记忆", memory_type="working", importance=0.5)
        mid_e = self.mm.add_memory(content="episodic记忆", memory_type="episodic", importance=0.5)

        self._assert(self.mm.find_memory(mid_w), "在working中找到记忆")
        self._assert(self.mm.find_memory(mid_e), "在episodic中找到记忆")

    def test_find_memory_not_exists(self):
        print_section("24. 查找不存在的记忆")
        self._reset()

        self._assert(not self.mm.find_memory("nonexistent_id"), "不存在的记忆返回False")

    # ==================== 8. 遗忘策略测试 ====================
    def test_forget_by_importance(self):
        print_section("25. 遗忘策略 - 重要性 (IMPORTANCE)")
        self._reset()

        self.mm.add_memory(content="高重要性", memory_type="working", importance=0.8)
        self.mm.add_memory(content="低重要性1", memory_type="working", importance=0.1)
        self.mm.add_memory(content="低重要性2", memory_type="working", importance=0.05)

        count = self.mm.forget_memory(strategy=ForgetType.IMPORTANCE.value, threshold=0.3)
        self._assert(count >= 2, f"遗忘了 {count} 条低重要性记忆 (期望>=2)")

        remaining = self.mm.get_all_memories()
        for m in remaining:
            self._assert(m.importance >= 0.3,
                          f"剩余记忆重要性≥0.3 ('{m.content[:10]}' = {m.importance})")

    def test_forget_by_time(self):
        print_section("26. 遗忘策略 - 时间 (TIME)")
        self._reset()

        # 添加一条 "旧" 记忆 (直接操作底层添加带旧时间戳的记忆)
        old_memory = MemoryItem(
            id=str(uuid.uuid4()), user_id=self.user_id,
            content="这是一条旧记忆",
            type="working",
            timestamp=datetime.now() - timedelta(days=60),
            importance=0.5, metadata={}
        )
        self.working_memory.add_memory(old_memory)

        new_memory = MemoryItem(
            id=str(uuid.uuid4()), user_id=self.user_id,
            content="这是一条新记忆",
            type="working",
            timestamp=datetime.now(),
            importance=0.5, metadata={}
        )
        self.working_memory.add_memory(new_memory)

        count = self.mm.forget_memory(strategy=ForgetType.TIME.value, max_age_days=30)
        self._assert(count >= 1, f"遗忘了 {count} 条过期记忆 (期望>=1)")

    def test_forget_by_capacity(self):
        print_section("27. 遗忘策略 - 容量 (CAPACITY)")
        self._reset()

        # max_capacity=10, 添加12条
        for i in range(12):
            self.mm.add_memory(
                content=f"容量测试记忆{i}",
                memory_type="working",
                importance=round(0.1 + i * 0.07, 2)
            )

        count = self.mm.forget_memory(strategy=ForgetType.CAPACITY.value)
        self._assert(count >= 2, f"容量遗忘了 {count} 条记忆 (期望>=2)")

    def test_forget_invalid_strategy(self):
        print_section("28. 遗忘策略 - 无效策略")
        self._reset()

        try:
            self.mm.forget_memory(strategy="invalid_strategy")
            self._assert(False, "应抛出 ValueError")
        except ValueError as e:
            self._assert(True, f"正确抛出 ValueError: {str(e)[:40]}")

    # ==================== 9. 获取所有记忆测试 ====================
    def test_get_all_memories(self):
        print_section("29. 获取所有记忆")
        self._reset()

        self.mm.add_memory(content="working1", memory_type="working", importance=0.5)
        self.mm.add_memory(content="working2", memory_type="working", importance=0.6)
        self.mm.add_memory(content="episodic1", memory_type="episodic", importance=0.7)

        all_memories = self.mm.get_all_memories()
        self._assert(len(all_memories) == 3, f"获取到 {len(all_memories)} 条记忆 (期望3)")

    def test_get_all_memories_empty(self):
        print_section("30. 获取所有记忆 - 空状态")
        self._reset()

        all_memories = self.mm.get_all_memories()
        self._assert(len(all_memories) == 0, "空状态返回空列表")

    # ==================== 10. 统计信息测试 ====================
    def test_get_memory_stats(self):
        print_section("31. 统计信息测试")
        self._reset()

        self.mm.add_memory(content="stats_working1", memory_type="working", importance=0.4)
        self.mm.add_memory(content="stats_working2", memory_type="working", importance=0.6)
        self.mm.add_memory(content="stats_episodic1", memory_type="episodic", importance=0.8)

        stats = self.mm.get_memory_stats()

        self._assert(stats["user_id"] == "test_user", f"stats.user_id={stats['user_id']}")
        self._assert("working" in stats["enabled_types"], "enabled_types 包含 working")
        self._assert("episodic" in stats["enabled_types"], "enabled_types 包含 episodic")
        self._assert(stats["total_memories"] == 3, f"stats.total_memories=3 (实际={stats['total_memories']})")
        self._assert("working" in stats["memories_by_type"], "memories_by_type 包含 working")
        self._assert("episodic" in stats["memories_by_type"], "memories_by_type 包含 episodic")
        self._assert("config" in stats, "stats 包含 config")
        self._assert(stats["config"]["max_capacity"] == 10, f"config.max_capacity=10")

    def test_get_memory_stats_empty(self):
        print_section("32. 统计信息 - 空状态")
        self._reset()

        stats = self.mm.get_memory_stats()
        self._assert(stats["total_memories"] == 0, "空状态: total_memories=0")
        self._assert("working" in stats["enabled_types"], "空状态仍包含 enabled_types")

    # ==================== 11. 清空记忆测试 ====================
    def test_clear_memories(self):
        print_section("33. 清空所有记忆")
        self._reset()

        self.mm.add_memory(content="待清空1", memory_type="working", importance=0.5)
        self.mm.add_memory(content="待清空2", memory_type="episodic", importance=0.5)
        self.mm.add_memory(content="待清空3", memory_type="perceptual", importance=0.5, metadata={"modality": "text"})

        self._assert(len(self.mm.get_all_memories()) >= 3, f"清空前有 {len(self.mm.get_all_memories())} 条记忆")

        self.mm.clear_memories()

        all_after = self.mm.get_all_memories()
        self._assert(len(all_after) == 0, f"清空后记忆数量=0 (实际={len(all_after)})")

    # ==================== 12. 合并记忆测试 ====================
    def test_merge_memories(self):
        print_section("34. 合并记忆 (working → episodic)")
        self._reset()

        self.mm.add_memory(content="高重要性工作记忆", memory_type="working", importance=0.9)
        self.mm.add_memory(content="低重要性工作记忆", memory_type="working", importance=0.2)
        self.mm.add_memory(content="中重要性工作记忆", memory_type="working", importance=0.6)

        merged = self.mm.merge_memories("working", "episodic", importance_threshold=0.5)
        self._assert(merged >= 2, f"合并了 {merged} 条记忆 (期望>=2, 重要性>=0.5)")

        # 验证 working 中高重要性的被移走
        working_remaining = self.mm.memory_types["working"].get_all_memories()
        episodic_all = self.mm.memory_types["episodic"].get_all_memories()

        self._assert(len(working_remaining) <= 1, f"working 剩余 {len(working_remaining)} 条 (低重要性)")
        self._assert(len(episodic_all) >= 2, f"episodic 有 {len(episodic_all)} 条 (含合并的)")

    def test_merge_invalid_type(self):
        print_section("35. 合并记忆 - 无效类型")
        self._reset()

        try:
            self.mm.merge_memories("working", "nonexistent")
            self._assert(False, "应抛出 ValueError")
        except ValueError as e:
            self._assert(True, f"正确抛出 ValueError: {str(e)[:40]}")

    def test_merge_no_match(self):
        print_section("36. 合并记忆 - 无匹配 (阈值极高)")
        self._reset()

        self.mm.add_memory(content="低重要性", memory_type="working", importance=0.1)

        merged = self.mm.merge_memories("working", "episodic", importance_threshold=0.99)
        self._assert(merged == 0, f"高阈值无匹配, 合并数={merged}")

    # ==================== 13. 同步与加载测试 ====================
    def test_sync_memories(self):
        print_section("37. 同步记忆 (sync_memories)")
        self._reset()

        self.mm.add_memory(content="同步测试", memory_type="working", importance=0.5)
        self.mm.add_memory(content="同步测试episodic", memory_type="episodic", importance=0.6)

        # sync_memories 不应抛异常
        try:
            self.mm.sync_memories()
            self._assert(True, "sync_memories 执行成功")
        except Exception as e:
            self._assert(False, f"sync_memories 抛出异常: {e}")

    def test_load_memories(self):
        print_section("38. 加载记忆 (load_memories)")
        self._reset()

        # load_memories 不应抛异常
        try:
            self.mm.load_memories()
            self._assert(True, "load_memories 执行成功")
        except Exception as e:
            self._assert(False, f"load_memories 抛出异常: {e}")

    # ==================== 14. 多类型记忆交互测试 ====================
    def test_cross_type_workflow(self):
        print_section("39. 跨类型完整工作流")
        self._reset()

        # 添加各类型记忆
        mid_w = self.mm.add_memory(content="工作记忆: 正在编写测试", memory_type="working", importance=0.8)
        mid_e = self.mm.add_memory(content="情景记忆: 今天完成了代码", memory_type="episodic", importance=0.7)
        mid_p = self.mm.add_memory(content="感知记忆: 看到的文本", memory_type="perceptual", importance=0.5, metadata={"modality": "text"})

        # 所有记忆应可被找到
        self._assert(self.mm.find_memory(mid_w), "working 记忆可被找到")
        self._assert(self.mm.find_memory(mid_e), "episodic 记忆可被找到")
        self._assert(self.mm.find_memory(mid_p), "perceptual 记忆可被找到")

        # 获取所有记忆
        all_memories = self.mm.get_all_memories()
        self._assert(len(all_memories) == 3, f"共有3条记忆 (实际={len(all_memories)})")

        # 统计
        stats = self.mm.get_memory_stats()
        self._assert(stats["total_memories"] == 3, "统计: total=3")

        # 更新 working 记忆
        self.mm.update_memory(mid_w, content="更新后的工作记忆", importance=0.95)
        self._assert(self.mm.find_memory(mid_w), "更新后仍可找到")

        # 删除 episodic 记忆
        self.mm.remove_memory(mid_e)
        self._assert(not self.mm.find_memory(mid_e), "删除后找不到 episodic 记忆")

        # 最终检查
        all_final = self.mm.get_all_memories()
        self._assert(len(all_final) == 2, f"删除后剩余2条 (实际={len(all_final)})")

    # ==================== 15. 边界情况测试 ====================
    def test_empty_content(self):
        print_section("40. 边界 - 空内容")
        self._reset()

        mid = self.mm.add_memory(content="", memory_type="working", importance=0.5)
        self._assert(mid != "", "空内容也能添加成功")

    def test_special_characters(self):
        print_section("41. 边界 - 特殊字符")
        self._reset()

        special = "Hello! @#$%^&*() 你好世界 🌍🎉"
        mid = self.mm.add_memory(content=special, memory_type="working", importance=0.5)
        self._assert(mid != "", "特殊字符添加成功")

    def test_add_many_memories(self):
        print_section("42. 边界 - 批量添加多条记忆")
        self._reset()

        ids = []
        for i in range(20):
            mid = self.mm.add_memory(
                content=f"批量记忆第{i+1}条",
                memory_type="working",
                importance=round(0.1 + i * 0.04, 2)
            )
            ids.append(mid)

        self._assert(len(ids) == 20, f"成功添加20条 (实际={len(ids)})")
        all_m = self.mm.get_all_memories()
        self._assert(len(all_m) >= 20, f"get_all_memories 返回 {len(all_m)} 条")

    # ==================== 运行所有测试 ====================
    def run_all(self):
        tests = [
            # 1. 初始化
            self.test_init_with_all_types,
            self.test_init_only_working,
            self.test_init_none_enabled,
            # 2. 添加记忆
            self.test_add_working_memory,
            self.test_add_episodic_memory,
            self.test_add_perceptual_memory,
            self.test_add_invalid_type,
            # 3. 自动分类
            self.test_auto_classify_episodic,
            self.test_auto_classify_semantic,
            self.test_auto_classify_working,
            self.test_auto_classify_perceptual_by_metadata,
            self.test_auto_classify_by_metadata_type,
            # 4. 删除记忆
            self.test_remove_working_memory,
            self.test_remove_episodic_memory,
            self.test_remove_nonexistent,
            # 5. 搜索记忆
            self.test_search_single_type,
            self.test_search_multi_type,
            self.test_search_all_types,
            self.test_search_empty,
            # 6. 更新记忆
            self.test_update_working_memory,
            self.test_update_episodic_memory,
            self.test_update_nonexistent,
            # 7. 查找记忆
            self.test_find_memory_exists,
            self.test_find_memory_not_exists,
            # 8. 遗忘策略
            self.test_forget_by_importance,
            self.test_forget_by_time,
            self.test_forget_by_capacity,
            self.test_forget_invalid_strategy,
            # 9. 获取所有记忆
            self.test_get_all_memories,
            self.test_get_all_memories_empty,
            # 10. 统计信息
            self.test_get_memory_stats,
            self.test_get_memory_stats_empty,
            # 11. 清空
            self.test_clear_memories,
            # 12. 合并
            self.test_merge_memories,
            self.test_merge_invalid_type,
            self.test_merge_no_match,
            # 13. 同步/加载
            self.test_sync_memories,
            self.test_load_memories,
            # 14. 跨类型工作流
            self.test_cross_type_workflow,
            # 15. 边界情况
            self.test_empty_content,
            self.test_special_characters,
            self.test_add_many_memories,
        ]

        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                test_name = test_func.__name__
                print_fail(f"{test_name} 抛出异常", str(e))
                traceback.print_exc()
                self.failed += 1
                self.errors.append(f"{test_name}: {e}")

        # 最终统计
        print_section("测试结果汇总")
        total = self.passed + self.failed
        print(f"  总计: {total} 项")
        print(f"  {Colors.GREEN}通过: {self.passed}{Colors.END}")
        print(f"  {Colors.RED}失败: {self.failed}{Colors.END}")
        if self.errors:
            print(f"\n  {Colors.RED}失败项:{Colors.END}")
            for i, err in enumerate(self.errors, 1):
                print(f"    {i}. {err}")
        print()
        return self.failed == 0


if __name__ == "__main__":
    tester = TestMemoryManage()
    success = tester.run_all()
    sys.exit(0 if success else 1)
