import json
import os
import sys
from typing import Any

example_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(example_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(example_dir, ".env"))

from agent.BasicAgent import BasicAgent
from core import enable_logging
from core.Message import Message
from core.llm import EasyLLM
from Tool.builtin import CalculatorTool


def _history_entry_to_payload(entry: Any) -> dict[str, Any]:
    if isinstance(entry, Message):
        payload = entry.model_dump()
        payload["entry_class"] = type(entry).__name__
        return payload
    if isinstance(entry, dict):
        return {
            "entry_class": "dict",
            "payload": entry,
        }
    return {
        "entry_class": type(entry).__name__,
        "payload": str(entry),
    }


def print_history(agent: BasicAgent) -> None:
    print("\n\n===== INTERNAL HISTORY =====")
    for index, entry in enumerate(agent.get_history(), start=1):
        print(f"\n[{index}]")
        print(
            json.dumps(
                _history_entry_to_payload(entry),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


def print_thinking_history(agent: BasicAgent) -> None:
    print("\n\n===== THINKING HISTORY =====")
    print(
        json.dumps(
            agent.get_thinking_history(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def print_trace_history(agent: BasicAgent) -> None:
    print("\n\n===== TRACE HISTORY =====")
    print(
        json.dumps(
            agent.get_trace_history(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main() -> None:
    enable_logging("INFO")

    llm = EasyLLM()
    agent = BasicAgent(
        name="history-demo",
        llm=llm,
        enable_tool=True,
        verbose_thinking=True,
    )
    agent.add_tool(CalculatorTool())
    agent.clear_history()

    query = "请先简单说明你要做什么，然后调用工具计算 4^12 + 6*412，最后告诉我结果。"

    print("===== STREAM OUTPUT =====")
    final_result = agent.stream_invoke(query)
    print("\n\n===== FINAL RESULT =====")
    print(final_result)

    print_history(agent)
    print_thinking_history(agent)
    print_trace_history(agent)


if __name__ == "__main__":
    main()
