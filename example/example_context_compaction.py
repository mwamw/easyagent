import json
import os
import sys
from typing import Any

example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.BasicAgent import BasicAgent
from context.compressor.history import RuleBasedHistoryCompactor
from context.manager import ContextManager
from context.token.budget import TokenBudget
from context.token.counter import TokenCounter
from core.history import CanonicalBlock, CanonicalMessage
from core.llm import EasyLLM


class DummyLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"


def print_block(title: str, payload: Any) -> None:
    print(f"\n===== {title} =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def message(role: str, content: str) -> CanonicalMessage:
    return CanonicalMessage(
        role=role,
        content=[CanonicalBlock(type="text", text=content)],
    )


def main() -> None:
    llm = DummyLLM()
    manager = ContextManager(
        max_tokens=260,
        budget=TokenBudget(max_tokens=260),
    )
    manager.builder._counter = TokenCounter(chars_per_token=1.0)
    manager.set_history_compactor(
        RuleBasedHistoryCompactor(token_counter=manager.builder.counter)
    )

    agent = BasicAgent(
        name="context-demo",
        llm=llm,
        system_prompt="你是一个负责上下文压缩演示的助手。",
    ).with_context(manager)

    agent.history = [
        message("user", "用户提出了第一轮非常长的需求说明，需要系统记住项目背景和目标。"),
        message("assistant", "助手确认了第一轮需求，并详细解释了计划和限制条件。"),
        message("user", "用户继续补充第二轮长约束，包括工具调用和恢复要求。"),
        message("assistant", "助手总结第二轮长约束，并给出后续实现顺序。"),
        message("user", "用户第三轮继续增加新的限制条件，并要求保持时间顺序和工具链完整。"),
        message("assistant", "助手第三轮记录新的限制条件，并说明不能直接粗暴截断历史。"),
        message("user", "用户第四轮要求会话恢复后继续复用摘要缓存。"),
        message("assistant", "助手第四轮确认恢复时需要保留压缩状态。"),
        message("user", "u5"),
        message("assistant", "a5"),
    ]
    original_history = [
        message.model_dump(mode="json") if hasattr(message, "model_dump") else message
        for message in agent.history
    ]

    query = "继续刚才的话题，并解释这次上下文是怎么被压缩的。"
    messages = agent._build_start_messages(query)

    print_block("ORIGINAL HISTORY BEFORE COMPACTION", original_history)
    print_block(
        "COMPACTED HISTORY SAVED ON AGENT",
        [
            message.model_dump(mode="json") if hasattr(message, "model_dump") else message
            for message in agent.history
        ],
    )
    print(f"\n===== COMPACTION FLAG =====\nwas_compacted={manager.last_history_was_compacted}")
    print_block("LAST COMPACTED HISTORY", manager.last_compacted_history)
    print_block("FINAL MESSAGES", messages)
    print(
        "\n===== TOKEN COUNTS =====\n"
        f"final_messages_tokens={manager.builder._counter.count_messages(messages)}\n"
        f"budget={manager.budget.max_tokens}"
    )


if __name__ == "__main__":
    main()
