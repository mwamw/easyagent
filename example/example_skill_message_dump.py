import os
import sys
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from core.Message import ToolMessage
from core.llm import EasyLLM
from skill.meta_tools import MetaSkill, SkillTool
from skill.registry import SkillRegistry


REAL_SKILLS_DIR = os.path.join(ROOT, "test", "real_skills")


class SpyEasyLLM(EasyLLM):
    def __init__(self):
        self.provide = "mock"
        self.provider_name = "mock"
        self.model = "mock-model"
        self.last_messages: list[Any] = []

    def invoke(self, messages, temperature: Optional[float] = None, **kwargs):
        self.last_messages = messages
        return "ok"

    def think(self, messages, temperature: Optional[float] = None):
        yield "ok"


def dump_messages(title: str, messages: list[Any]) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    for index, message in enumerate(messages, start=1):
        role = getattr(message, "role", None) or message.get("role", "unknown")
        content = getattr(message, "content", None) or message.get("content", "")
        print(f"\n[{index}] role={role}")
        print(content)


def main() -> None:
    SkillRegistry.reset()
    registry = SkillRegistry.instance()
    discovered = registry.discover_from_directory(REAL_SKILLS_DIR)
    if "crypto_skill" not in discovered:
        raise RuntimeError(f"未发现 crypto_skill，实际发现: {discovered}")

    agent = BasicAgent(name="message-dump-demo", llm=SpyEasyLLM())
    agent.with_skill(MetaSkill(registry, agent.skill_manager))

    query = "请帮我计算 'Autonomous Mode' 的 SHA-256。"

    # 1. 起始消息：只会看到 skill policy + listing，不会看到 crypto_skill 正文
    start_messages = agent._build_start_messages(query)
    dump_messages("A. 调用 skill_tool 之前的起始消息", start_messages)

    # 2. 模拟模型选择 crypto_skill，并调用 skill_tool
    skill_tool = SkillTool(registry, agent.skill_manager, set())
    tool_result = skill_tool.run({"skill_name": "crypto_skill"})

    # 3. 复现同轮后续推理的大致消息形态：
    #    - 原始 start messages
    #    - 一条 tool result
    #    - 一条 runtime skill context meta user message
    after_skill_messages = list(start_messages)
    after_skill_messages.append(
        ToolMessage(
            content=tool_result,
            tool_call_id="call_demo_skill_tool",
            name="skill_tool",
        )
    )
    agent._append_runtime_skill_context_message(after_skill_messages)
    dump_messages("B. 调用 skill_tool 之后的消息", after_skill_messages)

    SkillRegistry.reset()


if __name__ == "__main__":
    main()
