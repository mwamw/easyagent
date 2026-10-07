"""Compile and run a real .ea file, without credentials or network requests."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from easyagent.configuration import Options, default_registry
from easyagent.language import compile_file
from example.ui.offline_demo import OfflineLLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--query", help="Run one streaming request without a TTY")
    args = parser.parse_args()
    registry = default_registry()
    registry.register_model("offline", config_model=Options, factory=lambda options, paths: OfflineLLM(
        "这是 **EasyAgent Language 离线演示**，没有调用模型 API。\n\n"
        "当前 Agent 由 offline.ea 中的带类型模板装配，运行时仍是普通 BasicAgent。\n\n"
        "```python\nprogram = compile_file('offline.ea', registry=registry)\n"
        "agent = program.build('Demo')\n```\n\n"
        "在交互模式中可使用 /help 查看帮助、/exit 退出。"
    ))
    program = compile_file(Path(__file__).with_name("offline.ea"), registry=registry)
    if args.check:
        print(program.explain())
    elif args.query is not None:
        async def once():
            agent = program.build()
            try:
                async for event in agent.astream(args.query):
                    if event.type.value == "text_delta":
                        print(event.content, end="", flush=True)
                print()
            finally:
                await agent.aclose()
        asyncio.run(once())
    else:
        async def interactive():
            ui = program.create_ui()
            agent = None
            try:
                agent = program.build()
                await ui.run(agent)
            finally:
                try:
                    await ui.aclose()
                finally:
                    if agent is not None:
                        await agent.aclose()
        asyncio.run(interactive())


if __name__ == "__main__":
    main()
