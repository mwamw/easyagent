"""
MemoryTool 综合测试程序
覆盖:
  - 基本环境初始化 (LLM + Agent + ToolRegistry + MemoryTool)
  - 1. add 记忆
  - 2. stats 取统计信息
  - 3. search 搜索记忆
  - 4. get 获取完整记忆
  - 5. update 更新记忆
  - 6. consolidate 整合记忆
  - 7. forget 遗忘记忆
  - 8. clear 清空记忆
"""
import sys
import os
import traceback
import json
from datetime import datetime
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load context .env

_v2_path = os.path.join(_project_root, 'memory', 'V2')
if _v2_path not in sys.path:
    sys.path.insert(0, _v2_path)

from memory.V2.MemoryManage import MemoryManage
from memory.V2.BaseMemory import MemoryConfig
from memory.V2.WorkingMemory import WorkingMemory
from memory.V2.EpisodicMemory import EpisodicMemory
from Tool.builtin.memorytool import MemoryTool
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from memory.V2.Store.SQLiteDocumentStore import SQLiteDocumentStore
from memory.V2.Store.QdrantVectorStore import QdrantVectorStore
from memory.V2.Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel

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

class TestMemoryToolWithAgent:
    def __init__(self):
        print_section("初始化测试环境")
        self.passed = 0
        self.failed = 0
        self.errors = []
        
        # 为了测试 consolidate 和多类型支持，我们也初始化 episodic
        self.config = MemoryConfig(max_capacity=20)
        self.working_memory = WorkingMemory(self.config)
        
        # 简单实例化 episodic 依赖
        self.embedding_model = HuggingfaceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.episodic_doc_store = SQLiteDocumentStore(db_path="/tmp/test_tool_episodic.db")
        self.episodic_vector_store = QdrantVectorStore(way="memory", collection_name="test_tool_ep", vector_size=384)
        self.episodic_memory = EpisodicMemory(
            config=self.config,
            document_store=self.episodic_doc_store,
            vector_store=self.episodic_vector_store,
            embedding_model=self.embedding_model
        )
        
        self.mm = MemoryManage(
            config=self.config,
            user_id="test_tool_user",
            enable_working=True,
            working_memory=self.working_memory,
            enable_episodic=True,
            episodic_memory=self.episodic_memory,
            enable_semantic=False,
            enable_perceptual=False
        )
        
        self.memory_tool = MemoryTool(memory_manage=self.mm)
        print(f"tool description:{self.memory_tool.description}")
        print(f"tool parm:{self.memory_tool.get_openai_schema()}")
        self.tool_registry = ToolRegistry()
        self.tool_registry.registerTool(self.memory_tool)
        
        self.llm = EasyLLM() # Assume valid config internally
        
        # 带有 ToolRegistry 的 agent
        self.agent = BasicAgent(
            name="MemoryAgent",
            llm=self.llm,
            system_prompt="你是一个可以通过调用 memory_tool 来管理记忆的智能助手。当有管理需要的请求时，请准确调用 memory_tool 并使用其参数完成任务。尽量直接使用工具，不要过多废话。",
            enable_tool=True,
            tool_registry=self.tool_registry,
            verbose_thinking=False
        )
        
        self._memory_id_cache = None
        print_pass("测试环境初始化完成")

    def _assert(self, condition, msg, detail=""):
        if condition:
            print_pass(msg, detail)
            self.passed += 1
        else:
            print_fail(msg, detail)
            self.failed += 1
            self.errors.append(f"{msg}" + (f" ({detail})" if detail else ""))

    def test_llm_add_memory(self):
        print_section("1. LLM 调用 add 添加工作记忆")
        self.agent.history.clear()
        self.mm.clear_memories()
        
        prompt = "请使用 memory_tool 添加一条 working 类型记忆，内容是：'我的幸运数字是 42'，重要性为 0.9。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        memories = self.mm.get_all_memories()
        self._assert(len(memories) >= 1, "成功通过 LLM 存储了记忆", f"当前共有 {len(memories)} 条")
        if memories:
            self._memory_id_cache = memories[0].id
            print(f"提取出缓存记忆ID: {self._memory_id_cache}")

    def test_llm_stats_memory(self):
        print_section("2. LLM 调用 stats 获取统计")
        self.agent.history.clear()
        
        prompt = "请调用 memory_tool 获取当前的记忆系统统计信息。然后告诉我总共有多少条记忆。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        self._assert("1" in response or "一" in response, "LLM 能够获取统计并正确回答记忆数量", f"回答: {response}")

    def test_llm_search_memory(self):
        print_section("3. LLM 调用 search 搜索记忆")
        self.agent.history.clear()
        
        prompt = "请使用 memory_tool 搜索内容包含 '幸运数字' 的记忆，并直接告诉我我的幸运数字是多少。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        self._assert("42" in response, "LLM 成功搜索了记忆并提取了信息", f"回答: {response}")

    def test_llm_get_memory(self):
        print_section("4. LLM 调用 get 提取完整记忆")
        if not self._memory_id_cache:
            self._assert(False, "跳过: 缺少 cached memory_id")
            return
            
        self.agent.history.clear()
        prompt = f"请使用 memory_tool 的 get 操作，传入 memory_ids=[\"{self._memory_id_cache}\"] 这个ID来获取记忆内容，并将内容复述告诉我。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        self._assert("幸运数字" in response and "42" in response, "LLM 成功通过 ID 获取了完整记忆内容")

    def test_llm_update_memory(self):
        print_section("5. LLM 调用 update 更新记忆")
        if not self._memory_id_cache:
            self._assert(False, "跳过: 缺少 cached memory_id")
            return
            
        self.agent.history.clear()
        prompt = f"请使用 memory_tool 的 update 操作，将 memory_id 为 {self._memory_id_cache} 的记忆内容修改为 '我的幸运数字是 100'。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        memories = self.mm.get_all_memories()
        updated_content = [m.content for m in memories if m.id == self._memory_id_cache][0]
        self._assert("100" in updated_content, "LLM 成功通过 ID 更新了记忆的内容", f"当前内容: {updated_content}")

    def test_llm_consolidate_memory(self):
        print_section("6. LLM 调用 consolidate 整合记忆")
        self.agent.history.clear()
        self.mm.add_memory(content="待转化的高价值记忆", memory_type="working", importance=0.8)
        
        prompt = "请使用 memory_tool 的 consolidate 操作，将 working 类型的记忆整合到 episodic 类型的记忆中，重要性阈值设为 0.5。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        ep_memories = self.mm.memory_types["episodic"].get_all_memories()
        self._assert(len(ep_memories) > 0, "LLM 成功调用整合操作，将 working 转化为 episodic")

    def test_llm_remove_memory(self):
        print_section("7. LLM 调用 remove 删除记忆")
        if not self._memory_id_cache:
            self._assert(False, "跳过: 缺少 cached memory_id")
            return
            
        self.agent.history.clear()
        prompt = f"请使用 memory_tool 的 remove 操作，删除 memory_id 为 {self._memory_id_cache} 的这条记忆。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        self._assert(not self.mm.find_memory(self._memory_id_cache), "LLM 成功删除了指定的记忆")

    def test_llm_forget_memory(self):
        print_section("8. LLM 调用 forget 遗忘记忆")
        self.agent.history.clear()
        self.mm.add_memory(content="极低级别记忆", memory_type="working", importance=0.01)
        self.mm.add_memory(content="中等级别记忆", memory_type="working", importance=0.5)
        
        prompt = "请使用 memory_tool 的 forget 操作，对系统使用 'importance'（即重要性）策略进行遗忘，遗忘阈值 threshold 设为 0.3。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        memories = self.mm.get_all_memories()
        all_passed = all(m.importance >= 0.3 for m in memories)
        self._assert(all_passed, "LLM 成功调用遗忘策略清除了极低重要性的记忆")

    def test_llm_clear_memory(self):
        print_section("9. LLM 调用 clear 清空记忆")
        self.agent.history.clear()
        
        prompt = "请使用 memory_tool 的 clear 操作清空目前系统中所有的记忆。"
        print(f"用户输入: {prompt}")
        response = self.agent.invoke(prompt)
        print(f"Agent回答: {response}")
        
        memories = self.mm.get_all_memories()
        self._assert(len(memories) == 0, "LLM 成功执行了清空所有记忆的操作")

    def run_all(self):
        tests = [
            self.test_llm_add_memory,
            self.test_llm_stats_memory,
            self.test_llm_search_memory,
            self.test_llm_get_memory,
            self.test_llm_update_memory,
            self.test_llm_consolidate_memory,
            self.test_llm_remove_memory,
            self.test_llm_forget_memory,
            self.test_llm_clear_memory,
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
        print_section("MemoryTool (LLM 调用) 测试结果汇总")
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
    tester = TestMemoryToolWithAgent()
    success = tester.run_all()
    sys.exit(0 if success else 1)
