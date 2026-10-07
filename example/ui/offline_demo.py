"""Run a real terminal UI without credentials: python example/ui/offline_demo.py."""
import asyncio
from types import SimpleNamespace

from easyagent import BasicAgent, EasyLLM
from easyagent.ui import TerminalUI


class OfflineProvider:
    def __init__(self, text: str | None = None):
        self.text = text

    def build_request(self, messages, **kwargs):
        return {"messages": list(messages), **kwargs}

    def build_tool_payload(self, tools):
        return list(tools)

    def apply_cache_policy(self, request, request_input):
        return request

    async def async_stream_raw(self, request):
        async def chunks():
            text = self.text or (
                "这是一个 **离线终端演示**，没有调用模型 API。\n\n"
                "你可以继续输入、使用 Alt+Enter 换行，或者用 Ctrl+C 请求停止。\n\n"
                "```python\nagent = load_agent('agent.yaml')\nawait ui.run(agent)\n```\n\n"
                "使用 /help 查看命令，使用 /exit 退出。"
            )
            for start in range(0, len(text), 3):
                await asyncio.sleep(0.04)
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=text[start:start + 3], tool_calls=None,
                                          reasoning_content=None, reasoning=None),
                    finish_reason=None,
                )])
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None,
                                      reasoning_content=None, reasoning=None),
                finish_reason="stop",
            )])
        return chunks()


class OfflineLLM(EasyLLM):
    def __init__(self, text: str | None = None):
        self.provider_name = "openai"
        self.model = "offline-demo"
        self.max_tokens = 256
        self.temperature = 0
        self.kwargs = {}
        self._provider = OfflineProvider(text)
        self.client = None


async def main():
    agent = BasicAgent("offline-demo", OfflineLLM())
    ui = TerminalUI()
    try:
        await ui.run(agent)
    finally:
        await ui.aclose()
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
