from __future__ import annotations

import os
import tempfile
import sys
example_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path to allow importing easyagent
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")
from pydantic import BaseModel

from easyagent import (
    BasicAgent,
    EasyLLM,
    InMemoryObservabilityStore,
    ObservabilityManager,
    SessionStore,
    Tool,
    ToolRegistry,
)


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    def __init__(self):
        super().__init__(
            name="EchoTool",
            description=(
                "Echo the provided text back to the agent. "
                "Use this tool when you need a deterministic local tool call in order to "
                "validate tool observability, tool metrics, and trace summaries."
            ),
            parameters=EchoParams,
            read_only=True,
            side_effect_level="none",
            resource_scope=["runtime"],
        )

    def run(self, parameters: dict):
        return f"echo:{parameters['text']}"


def build_llm() -> EasyLLM:
    return EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    return registry


def main() -> None:
    llm = build_llm()
    registry = build_registry()

    agent = BasicAgent(
        name="observability-demo",
        llm=llm,
        system_prompt=(
            "You are validating EasyAgent observability. "
            "Use tools when the task explicitly asks for a local deterministic action. "
            "When you finish, keep the answer concise."
        ),
    ).with_tool(registry).with_observability()

    # 1. 普通调用：记录 plain agent run + llm request
    plain_result = agent.invoke("用一句话说明当前框架为什么需要 observability。")
    print("plain_result:", plain_result)

    # 2. 工具调用：记录 tool agent run + llm requests + tool execution
    tool_result = agent.invoke("调用 EchoTool，参数 text='phase-i-observability'，然后总结它对观测的意义。")
    print("tool_result:", tool_result)

    # 3. 流式调用：记录 stream=true 的 plain 观测事件
    stream_events = list(agent.stream("流式输出一句总结，说明 observability summary 能提供什么。"))
    print("stream_events:", stream_events)

    # 4. 读取观测聚合视图
    assert agent.observability is not None
    summary = agent.observability.summary()
    recent_invokes = [record.model_dump(mode="json") for record in agent.observability.list()[-10:]]
    latest_trace = agent.observability.latest().trace if agent.observability.latest() else []

    print("observability_summary:")
    print(summary)
    print("recent_invokes:")
    print(recent_invokes)
    print("latest_trace:")
    print(latest_trace)

    # 5. 保存并恢复 session，验证 observability_state 跟随会话恢复
    with tempfile.TemporaryDirectory() as tempdir:
        store = SessionStore(os.path.join(tempdir, "observability_example.db"))
        agent.save_session("phase-i-observability", store=store)

        restored = BasicAgent.load_session(
            "phase-i-observability",
            llm=build_llm(),
            store=store,
            observability_manager=ObservabilityManager(InMemoryObservabilityStore()),
        )
        try:
            assert restored.observability is not None
            restored_summary = restored.observability.summary()
            restored_invokes = restored.observability.list()

            print("restored_summary:")
            print(restored_summary)
            print("restored_invokes:")
            print(restored_invokes)
        finally:
            restored.close()

    agent.close()


if __name__ == "__main__":
    main()
