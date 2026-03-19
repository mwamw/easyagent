import sys
import os
import time
import uuid

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_v2_path = os.path.join(_project_root, 'memory', 'V2')
if _v2_path not in sys.path:
    sys.path.insert(0, _v2_path)

# Setup context
import context
from dotenv import load_dotenv
load_dotenv()

from memory.V2.MemoryManage import MemoryManage
from memory.V2.BaseMemory import MemoryConfig
from memory.V2.WorkingMemory import WorkingMemory
from memory.V2.EpisodicMemory import EpisodicMemory
from memory.V2.PerceptualMemory import PerceptualMemory
from memory.V2.SemanticMemory import SemanticMemory
from memory.V2.Store.Neo4jGraphStore import Neo4jGraphStore
from memory.V2.Extractor.Extractor import Extractor
from Tool.builtin.memorytool import register_memory_tools
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from memory.V2.Store.SQLiteDocumentStore import SQLiteDocumentStore
from memory.V2.Store.QdrantVectorStore import QdrantVectorStore
from memory.V2.Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel

class TestAgentMemoryIntegration:
    def __init__(self):
        print("========== 初始化测试环境 ==========")
        self.config = MemoryConfig(max_capacity=20)
        
        # 使用基于内存的 SQLite 和 Qdrant 以防无相关环境依赖
        self.embedding_model = HuggingfaceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.working_memory = WorkingMemory(self.config, self.embedding_model)
        
        self.episodic_doc_store = SQLiteDocumentStore(db_path=":memory:")
        self.episodic_vector_store = QdrantVectorStore(way="memory", collection_name="test_ep_integration")
        self.episodic_memory = EpisodicMemory(
            config=self.config,
            document_store=self.episodic_doc_store,
            vector_store=self.episodic_vector_store,
            embedding_model=self.embedding_model
        )
        # 简单实例化 perceptual 依赖
        self.perceptual_doc_store = SQLiteDocumentStore(db_path="/tmp/test_tool_perceptual.db")
        self.perceptual_vector_store = QdrantVectorStore(way="memory", collection_name="test_tool_pe", vector_size=384)
        self.perceptual_image_store = QdrantVectorStore(way="memory", collection_name="test_tool_pe_image", vector_size=512)
        self.perceptual_memory = PerceptualMemory(
            memory_config=self.config,
            document_store=self.perceptual_doc_store,
            vector_stores={"text": self.perceptual_vector_store, "image": self.perceptual_image_store},
            embedding_model=self.embedding_model,
            supported_modalities=["text", "image"]
        )

        # 简单实例化 semantic 依赖
        # self.graph_store = Neo4jGraphStore(uri="bolt://localhost:17687", username="neo4j", password="password")
        # self.vector_store = QdrantVectorStore(way="memory", collection_name="semantic_memory")
        self.llm = EasyLLM() # Assume valid config internally
        # self.extractor = Extractor(self.llm, False)
        # self.semantic_memory = SemanticMemory(
        #     memory_config=MemoryConfig(),
        #     vector_store=self.vector_store,
        #     graph_store=self.graph_store,
        #     extractor=self.extractor,
        #     embedding_model=self.embedding_model
        # )
        # 初始化 MemoryManage，为了简单只开启 working 和 episodic
        self.mm = MemoryManage(
            config=self.config,
            user_id="test_integration_user",
            enable_working=True,
            working_memory=self.working_memory,
            enable_episodic=True,
            episodic_memory=self.episodic_memory,
            enable_semantic=False,
            # semantic_memory=self.semantic_memory,
            enable_perceptual=True,
            perceptual_memory=self.perceptual_memory
        ) 
        
        self.tool_registry = ToolRegistry()
        
        # 这里验证 Agent 是否能和 MemoryManage 自动结合（.with_memory或初始化传入）
        self.agent = BasicAgent(
            name="IntegrationTestAgent",
            llm=self.llm,
            system_prompt="你是一个具有记忆系统的测试AI。请按照用户的指令进行操作。",
            enable_tool=True,
            tool_registry=self.tool_registry,
            verbose_thinking=False
        ).with_memory(self.mm)
        
        # .with_memory 应该自动注册了工具
        tools = self.tool_registry.get_openai_tools()
        
        
        assert any(t["function"]["name"] == "add_memory_tool" for t in tools), "Tool registry failed to auto-register memory tools!"
        
        from context.source.memory_source import MemoryContextSource
        from context.manager import ContextManager
        context_manager = ContextManager(max_tokens=50000, auto_history=True)
        self.agent.with_context(context_manager)
        print("测试环境初始化完成！")

    def test_implicit_prompt_injection(self):
        print("\n========== 测试1: 潜意识记忆注入 ==========")
        self.agent.clear_history()
        self.mm.clear_memories()
        
        # 我们手动向 Working Memory 添加一条便签
        self.mm.add_memory(content="系统的核心密码是：42", memory_type="working", importance=1.0)
        
        # 问 Agent 这个信息，Agent 虽然没有历史，但是 system prompt 里有它的 inject 
        prompt = "系统的核心密码是多少？不要调用任何工具，只能依靠你的已有记忆。"
        print(f"User: {prompt}")
        
        response = self.agent.invoke(prompt)
        print(f"Agent: {response}")
        
        if "42" in response:
            print("✅ 潜意识记忆注入测试通过！")
        else:
            print("❌ 潜意识记忆注入测试失败！")

    def test_agent_tool_usage_for_memory(self):
        print("\n========== 测试2: 智能调用工具读写记忆 ==========")
        self.agent.clear_history()
        self.mm.clear_memories()
        
        # 不再明确指示要求使用工具和指定记忆类型，期待 Agent 自动判断
        prompt1 = "我告诉你一个重要的秘密：昨天小明给了我一个苹果。这对我很重要，请帮我记住这件事。"
        print(f"User: {prompt1}")
        res1 = self.agent.invoke(prompt1)
        print(f"Agent: {res1}")
        
        # 检查是否确实存入了 episodic 或 working
        ep_memories = self.mm.memory_types["episodic"].get_all_memories()
        working_memories = self.mm.memory_types["working"].get_all_memories()
        assert len(ep_memories) > 0 or len(working_memories) > 0, "Agent未能成功调用工具存入记忆"
        print("✅ 成功发现智能存入的新记忆")
        
        # 触发搜索
        self.agent.clear_history()
        prompt2 = "昨天小明给了我什么东西？"
        print(f"User: {prompt2}")
        res2 = self.agent.invoke(prompt2)
        print(f"Agent: {res2}")
        
        if "苹果" in res2:
            print("✅ 智能检索记忆测试通过！")
        else:
            print("❌ 智能检索记忆测试失败！")
            
    def test_memory_update_and_remove(self):
        print("\n========== 测试4: 智能调用工具更新和删除记忆 ==========")
        self.agent.clear_history()
        self.mm.clear_memories()
        
        prompt1 = "我最喜欢的水果是香蕉，请一定要记住。"
        print(f"User: {prompt1}")
        self.agent.invoke(prompt1)
        
        prompt2 = "我口味变了，我现在最喜欢的水果是苹果，不再是香蕉了，请更新你的记忆。"
        print(f"User: {prompt2}")
        res2 = self.agent.invoke(prompt2)
        print(f"Agent: {res2}")
        
        self.agent.clear_history()
        prompt3 = "我现在最喜欢的水果是什么？"
        res3 = self.agent.invoke(prompt3)
        print(f"Agent: {res3}")
        
        if "苹果" in res3 and "香蕉" not in res3:
            print("✅ 记忆更新测试通过！")
        else:
            print("❌ 记忆更新测试不完全通过或失败！答案中包含: " + res3)
            
        prompt4 = "刚才说的关于我最喜欢吃什么水果的事情，全部忘掉吧，把那条记忆删了。"
        print(f"User: {prompt4}")
        self.agent.invoke(prompt4)
        
        self.agent.clear_history()
        prompt5 = "我最喜欢吃什么水果？"
        res5 = self.agent.invoke(prompt5)
        print(f"Agent: {res5}")
        
        # Agent 应该回答不知道或没有相关记录
        if "苹果" not in res5 and "香蕉" not in res5:
            print("✅ 记忆删除测试通过！")
        else:
            print("❌ 记忆删除测试失败！答案中仍有相关信息: " + res5)
            
    def test_background_auto_extraction(self):
        print("\n========== 测试3: 后台自动记忆提炼机制 ==========")
        self.agent.clear_history()
        self.mm.clear_memories()
        
        current_episodic_count = len(self.mm.memory_types["episodic"].get_all_memories())
        current_working_count = len(self.mm.memory_types["working"].get_all_memories())
        
        # 模拟频繁对话，达到 _unextracted_msg_count >= 5 阈值
        print("开始输入 5 轮闲聊对话...")
        for i in range(5):
            msg = f"这是第 {i+1} 轮闲聊内容，我很喜欢吃水果，特别是香蕉"
            print(f"User: {msg}")
            # 注意: add_message 或者 add_user_message 都会触发检查
            self.agent.add_user_message(msg)
            self.agent.add_assistant_message(f"好的，我收到了你的第 {i+1} 轮消息。")
            
        print("等待后台提炼线程执行 (约7秒)...")
        time.sleep(7) # 预留时间给守护线程执行 LLM
        
        # 验证是否有新的记忆被自动注入到系统中
        new_episodic_count = len(self.mm.memory_types["episodic"].get_all_memories())
        new_working_count = len(self.mm.memory_types["working"].get_all_memories())
        
        total_new_memories = (new_episodic_count - current_episodic_count) + (new_working_count - current_working_count)
        
        if total_new_memories > 0:
            print(f"✅ 后台自动提炼机制测试通过！新增了 {total_new_memories} 条记忆。")
            # 打印这些新记忆
            print(self.mm.get_all_memories())
        else:
            print("❌ 未观察到自动提炼产生新的记忆，可能是LLM抽取失败或线程未执行。")

    def test_working_memory_management(self):
        print("\n========== 测试5: Working Memory 的自动管理 ==========")
        #测试自动将复杂任务中的关键信息保存到 Working Memory，并在后续对话中正确利用这些信息。
        self.agent.clear_history()
        self.mm.clear_memories()
        prompt1 = "我正在计划一个秘密派对，地点在海边别墅。"
        print(f"User: {prompt1}")
        self.agent.invoke(prompt1)
        self.agent.clear_history()
        prompt2 = "请帮我回忆一下这个秘密派对的地点在哪里？"
        print(self.agent._build_start_messages(query=prompt2))
        print(f"User: {prompt2}")
        res2 = self.agent.invoke(prompt2)
        print(f"Agent: {res2}")
        self.agent.clear_history()
        
        #测试话题重大变更时，Agent 是否能正确更新 Working Memory 中的相关信息。
        prompt3 ="量子化学是研究分子和原子在量子力学框架下行为的学科。其中有哪些必须了解的核心概念？"
        print(f"User: {prompt3}")
        print(self.agent._build_start_messages(query=prompt3))

        res3 = self.agent.invoke(prompt3)
        print(f"Agent: {res3}")
    def run_all(self):
        try:
            # self.test_implicit_prompt_injection()
            self.test_working_memory_management()
            # self.test_agent_tool_usage_for_memory()
            # self.test_background_auto_extraction()
            # self.test_memory_update_and_remove()
            print("\n🏁 所有Agent-Memory集成测试结束！")
        except Exception as e:
            print(f"\n❌ 测试过程中发生异常: {e}")

if __name__ == "__main__":
    tester = TestAgentMemoryIntegration()
    tester.run_all()
