from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context import LLMHistoryCompactor
from context.token.counter import TokenCounter
from core.llm import EasyLLM


def build_llm() -> EasyLLM:
    return EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )


def build_canonical_history() -> list[dict]:
    return [
        {
            "record_type": "canonical_message",
            "role": "user",
            "provider": "openai",
            "provider_message_type": "user",
            "content": [
                {
                    "type": "text",
                    "text": "使用工具翻译下面的文字到英语并判断这个工具正确吗: 你是谁，在哪里，并帮我计算 3^22。",
                }
            ],
            "metadata": {},
        },
        {
            "record_type": "canonical_message",
            "role": "assistant",
            "provider": "openai",
            "provider_message_type": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "text": "我需要先调用翻译工具，再调用计算器，最后判断翻译工具是否真的完成了英文翻译。",
                },
                {
                    "type": "function_call",
                    "name": "translate_tool",
                    "call_id": "call_translate_1",
                    "arguments": {"target_lang": "en", "text": "你是谁，在哪里"},
                },
                {
                    "type": "function_call",
                    "name": "calculator",
                    "call_id": "call_calc_1",
                    "arguments": {"expression": "3**22"},
                },
            ],
            "metadata": {},
        },
        {
            "record_type": "canonical_message",
            "role": "tool",
            "provider": "openai",
            "provider_message_type": "tool",
            "content": [
                {
                    "type": "function_response",
                    "name": "translate_tool",
                    "call_id": "call_translate_1",
                    "output": "Translated: 你是谁，在哪里",
                }
            ],
            "metadata": {},
        },
        {
            "record_type": "canonical_message",
            "role": "tool",
            "provider": "openai",
            "provider_message_type": "tool",
            "content": [
                {
                    "type": "function_response",
                    "name": "calculator",
                    "call_id": "call_calc_1",
                    "output": "31381059609",
                }
            ],
            "metadata": {},
        },
        {
            "record_type": "canonical_message",
            "role": "assistant",
            "provider": "openai",
            "provider_message_type": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "翻译结果：Who are you, and where are you? "
                        "翻译工具这次没有真正返回英文，所以工具结果不正确。"
                        "计算结果：3^22 = 31381059609。"
                    ),
                }
            ],
            "metadata": {},
        },
    ]


def run_sync_test() -> None:
    print("== sync compact ==", flush=True)
    llm = build_llm()
    compactor = LLMHistoryCompactor(
        llm=llm,
        token_counter=TokenCounter(chars_per_token=1.0),
        recent_turns=1,
    )
    history = build_canonical_history()
    try:
        result = compactor.compact(history, max_tokens=120)
        print("sync result type:", type(result).__name__, flush=True)
        print("sync result size:", len(result), flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print("sync compact failed:", repr(exc), flush=True)
        raise


async def run_async_test() -> None:
    print("== async compact ==", flush=True)
    llm = build_llm()
    compactor = LLMHistoryCompactor(
        llm=llm,
        token_counter=TokenCounter(chars_per_token=1.0),
        recent_turns=1,
    )
    history = build_canonical_history()
    try:
        result = await compactor.acompact(history, max_tokens=120)
        print("async result type:", type(result).__name__, flush=True)
        print("async result size:", len(result), flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        print("async compact failed:", repr(exc), flush=True)
        raise


def main() -> None:
    run_sync_test()
    # asyncio.run(run_async_test())


if __name__ == "__main__":
    main()
