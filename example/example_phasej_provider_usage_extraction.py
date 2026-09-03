from __future__ import annotations
import os
import sys


example_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path to allow importing easyagent
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, "/home/wxd/LLM/EasyAgent")
"""
真实示例：演示 provider-first usage extraction。

这个 example 不会在自动化里执行，留给后续手动调试。

重点展示两条实际可跑的链路：
1. openai chat
2. openai_responses

二者都使用同一套本地 openai-compatible 服务配置。
"""

from agent import BasicAgent
from core.llm import EasyLLM
from observability import InMemoryObservabilityStore


OPENAI_COMPAT_CONFIG = {
    "provider": "openai",
    "base_url": "http://127.0.0.1:5124/v1",
    "api_key": "122",
    "model": "qwen3.5-9b",
}


def demo_openai_chat_usage() -> None:
    llm = EasyLLM(**OPENAI_COMPAT_CONFIG)

    raw_response = llm.invoke_raw(
        [
            {
                "role": "user",
                "content": "请用一句话介绍你自己。",
            }
        ]
    )
    usage = llm.extract_usage_metrics(raw_response)

    print("=== OpenAI Chat Raw Usage ===")
    print(usage)

    agent = BasicAgent(name="provider-usage-openai-chat", llm=llm).with_observability(
        store=InMemoryObservabilityStore(),
    )
    events = list(agent.stream("请再用两句话介绍一下你自己。"))

    print("=== OpenAI Chat Stream Events ===")
    print(events)
    assert agent.observability is not None
    latest = agent.observability.latest()
    print("=== OpenAI Chat Latest LLM Invoke ===")
    print(latest.llm_invokes[-1] if latest and latest.llm_invokes else None)
    print("=== OpenAI Chat Summary ===")
    print(agent.observability.summary())


def demo_openai_responses_usage() -> None:
    llm = EasyLLM(
        provider="openai_responses",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )

    raw_response = llm.invoke_raw(
        [
            {
                "role": "user",
                "content": "请用一句话介绍你自己。",
            }
        ]
    )
    usage = llm.extract_usage_metrics(raw_response)

    print("=== OpenAI Responses Raw Usage ===")
    print(usage)

    final_event = None
    for event in llm.stream_events(
        [
            {
                "role": "user",
                "content": "请流式输出一句简短介绍。",
            }
        ]
    ):
        if event.get("type") == "final_response":
            final_event = event

    print("=== OpenAI Responses Stream Final Event ===")
    print(final_event)
    if final_event is not None:
        print("=== OpenAI Responses Stream Usage ===")
        print(llm.extract_usage_metrics(final_event))


if __name__ == "__main__":
    demo_openai_chat_usage()
    demo_openai_responses_usage()
