import os
from re import A
import sys
from typing import Optional

from dotenv import load_dotenv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv()

from core.callbacks import CallbackManager
from orchestrator import (
    SequentialOrchestrator,
    SupervisorOrchestrator,
    GroupChatOrchestrator,
    OrchestrationError,
    AgentNotFoundError,
)

# 使用鸭子类型替代直接继承 BaseAgent，避免环境依赖问题
class MockAgent:
    """用于测试的 Mock Agent"""
    def __init__(self, name: str, response: str = "mock response", description: str = ""):
        self.name = name
        self.description = description
        self._response = response
        self._call_count = 0
        self.callback_manager = CallbackManager()
    
    def invoke(self, query: str, max_iter: int = 10, temperature: float = 0.7, **kwargs) -> str:
        self._call_count += 1
        return self._response

    def get_enhanced_prompt(self) -> str:
        return ""


class OrchestratorIntegrationRunner:
    def __init__(self):
        print("========== 初始化 Orchestrator 集成测试环境 ==========")
        # 只在实际需要时初始化 LLM，以防没有配置 key
        self.has_llm_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
        if self.has_llm_key:
            from core.llm import EasyLLM
            self.llm = EasyLLM()
            try:
                self.llm.invoke([{"role": "user", "content": "测试 LLM 连接"}])
                print("✅ EasyLLM 初始化完成（真实 LLM）")
            except Exception as e:
                print(f"❌ EasyLLM 初始化失败: {e}")
        else:
            self.llm = None
            print("⚠️ 未检测到 LLM 配置，将跳过真实测试")

    def test_mock_sequential(self):
        print("\n========== 测试1: SequentialOrchestrator (Mock) ==========")
        try:
            orch = SequentialOrchestrator(
                name="test_pipeline",
                pipeline=["step1", "step2", "step3"],
            )
            orch.add_agent("step1", MockAgent("step1", response="输出1"))
            orch.add_agent("step2", MockAgent("step2", response="输出2"))
            orch.add_agent("step3", MockAgent("step3", response="最终输出"))
            
            result = orch.run("初始输入")
            print("Sequential 最终结果 =>", result)
            assert result == "最终输出", "Sequential 结果异常"
            print("✅ SequentialOrchestrator (Mock) 测试通过")
        except Exception as e:
            print(f"❌ SequentialOrchestrator (Mock) 失败: {e}")

    def test_mock_groupchat(self):
        print("\n========== 测试2: GroupChatOrchestrator (Mock round_robin) ==========")
        try:
            # mock round_robin 不需要 moderator LLM 生成总结，但会做 fallback
            class DummyLLM:
                def invoke(self, *args, **kwargs):
                    return "Mock 讨论总结"
            
            orch = GroupChatOrchestrator(
                name="test_chat",
                moderator_llm=DummyLLM(), # type: ignore
                max_rounds=2,
                speaker_selection="round_robin"
            )
            orch.add_agent("A", MockAgent("A", response="A说的话"))
            orch.add_agent("B", MockAgent("B", response="B说的话"))
            
            result = orch.run("开始讨论")
            print("GroupChat 最终结果 =>", result)
            assert "Mock 讨论总结" in result, "GroupChat 结果异常"
            print("✅ GroupChatOrchestrator (Mock) 测试通过")
        except Exception as e:
            print(f"❌ GroupChatOrchestrator (Mock) 失败: {e}")

    def test_real_sequential_pipeline(self):
        print("\n========== 测试3: SequentialOrchestrator 真实 LLM (作家+翻译家) ==========")
        if not self.has_llm_key:
            print("⚠️ 跳过此真实 LLM 测试")
            return

        try:
            from agent import BasicAgent
            writer_agent = BasicAgent(
                name="Writer",
                llm=self.llm,
                system_prompt="你是一个极简主义诗人。请你总是用 3 行以内写出深刻的短诗。",
                description="擅长创作简短、深刻的诗歌"
            )
            translator_agent = BasicAgent(
                name="Translator",
                llm=self.llm,
                system_prompt="你是一个资深翻译家。请将收到的内容精确翻译为英文，不添加任何额外解释。",
                description="能够将文本翻译为英文"
            )
            
            orch = SequentialOrchestrator(
                name="诗歌创作流水线",
                pipeline=["Writer", "Translator"]
            )
            orch.add_agent("Writer", writer_agent).add_agent("Translator", translator_agent)
            
            query = "请以'星空'为主题写一首诗。"
            print("User:", query)
            result = orch.run(query)
            print("\n[Sequential 最终结果]:", result)
            assert len(result) > 0, "返回结果为空"
            query2="请以初恋为主题写一首诗。"
            print("User:", query2)
            result2 = orch.run(query2)
            print("\n[Sequential 最终结果]:", result2)
            assert len(result2) > 0, "返回结果为空"
            print("✅ SequentialOrchestrator 真实链路测试通过")
        except Exception as e:
            print(f"❌ 真实 SequentialOrchestrator 测试失败: {e}")

    def test_real_supervisor_orchestrator(self):
        print("\n========== 测试4: SupervisorOrchestrator 真实 LLM (动态调度工具) ==========")
        if not self.has_llm_key:
            print("⚠️ 跳过此真实 LLM 测试")
            return

        try:
            from agent import BasicAgent
            from Tool.builtin.calculator import CalculatorTool
            from Tool import ToolRegistry
            
            calc_tool = CalculatorTool()
            registry = ToolRegistry()
            registry.register_tool(calc_tool)
            
            math_agent = BasicAgent(
                name="MathWorker",
                llm=self.llm,
                system_prompt="你是一个数学家。碰到数学计算务必使用 calculator_tool，并将计算结果反馈。",
                description="擅长执行复杂数学运算步骤，带有精确计算器工具",
            ).with_tool(registry)
            logic_agent = BasicAgent(
                name="LogicWorker",
                llm=self.llm,
                system_prompt="你是一个常识推理家。你的职责是解析复杂的文字逻辑或提供常识事实。",
                description="擅长处理常识、推理和逻辑解析，但不做具体数值计算"
            )
            
            orch = SupervisorOrchestrator(
                name="解题主管",
                supervisor_llm=self.llm,
                max_rounds=5
            )
            orch.add_agent("MathWorker", math_agent)
            orch.add_agent("LogicWorker", logic_agent)
            
            query = "请问如果我每天工作 8 小时，工作 7 天，每小时能赚 123.45 元，我总共能赚多少钱？"
            print("User:", query)
            result = orch.run(query)
            print("\n[Supervisor 最终结果]:", result)
            
            # 断言最终数字（8 * 7 * 123.45 = 6913.2）
            assert "6913.2" in result, "计算结果异常"
            print("✅ SupervisorOrchestrator 真实链路测试通过")
        except Exception as e:
            print(f"❌ 真实 SupervisorOrchestrator 测试失败: {e}")

    def run_all(self):
        print("\n========== 开始执行 Orchestrator 全链路测试 ==========")
        # self.test_mock_sequential()
        # self.test_mock_groupchat()
        self.test_real_sequential_pipeline()
        # self.test_real_supervisor_orchestrator()
        print("\n🏁 全部 Orchestrator 集成测试执行完成")


def test_orchestrator_context():
    from orchestrator.context import SharedContext
    from orchestrator.message import AgentMessage,MessageType
    print("\n========== 测试 SharedContext 和 AgentMessage ==========")
    message1=AgentMessage(
        sender="agent1",
        receiver="agent2",
        content="这是一个测试消息",
        msg_type="task",
        metadata={"key": "value"}
    )
    print("AgentMessage:", message1)
    context = SharedContext()
    context.add(
        sender=message1.sender,
        receiver=message1.receiver,
        content=message1.content,
        msg_type=message1.msg_type,
        **message1.metadata
    )
    message2=AgentMessage(
        sender="agent2",
        receiver="agent1",
        content="这是回复消息",
        msg_type="result",
        metadata={"response_to": "message1"}
    )
    context.add_message(message2)
    print("SharedContext 消息列表:")
    for msg in context.messages:
        print(msg)
if __name__ == "__main__":
    runner = OrchestratorIntegrationRunner()
    runner.run_all()
    # test_orchestrator_context()
