import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM
from skill.meta_tools import SkillTool
from skill.registry import SkillRegistry


REAL_SKILLS_DIR = os.path.join(ROOT, "test", "real_skills")


class SpyEasyLLM(EasyLLM):
    def __init__(self):
        self.provide = "mock"
        self.provider_name = "mock"
        self.model = "mock-model"
        self.last_messages = []

    def invoke(self, messages, temperature=None, **kwargs):
        self.last_messages = messages
        return "ok"

    def think(self, messages, temperature=None):
        yield "ok"


def _make_agent() -> BasicAgent:
    return BasicAgent(name="crypto-skill-tester", llm=SpyEasyLLM())


def _extract_system_prompt(messages) -> str:
    assert messages, "messages 不应为空"
    first = messages[0]
    return getattr(first, "content", "") or first.get("content", "")


@pytest.fixture()
def registry():
    SkillRegistry.reset()
    registry = SkillRegistry.instance()
    discovered = registry.discover_from_directory(REAL_SKILLS_DIR)
    assert "crypto_skill" in discovered
    yield registry
    SkillRegistry.reset()


def test_crypto_skill_registered_directly_only_appears_in_listing(registry):
    """
    复现你现在的场景：

        sk = registry.create("crypto_skill")
        agent.skill_manager.register(sk)
        agent._build_start_messages("")

    预期：
    - 能在 skill listing 里看到 `crypto_skill`
    - 但看不到它的正文 `## 密码学能力`
    """
    agent = _make_agent()

    sk = registry.create("crypto_skill")
    agent.skill_manager.register(sk)

    messages = agent._build_start_messages("")
    system_prompt = _extract_system_prompt(messages)

    assert "## 可用 Skills" in system_prompt
    assert "`crypto_skill`" in system_prompt
    assert "## 密码学能力" not in system_prompt


def test_crypto_skill_enters_runtime_context_after_skill_tool(registry):
    """
    on-demand skill 的正文不会常驻。
    调用 skill_tool 后，正文会进入 runtime skill context。
    """
    agent = _make_agent()
    tool = SkillTool(registry, agent.skill_manager, set())

    result = tool.run({"skill_name": "crypto_skill"})
    text = result.to_display_string()
    assert result.status == "success"
    assert "已注入 Skill `crypto_skill`" in text
    assert "详细正文已注入当前 invoke 的后续推理链" in text
    assert "<skill>" not in text

    injected_messages = []
    agent._append_runtime_skill_context_message(injected_messages)

    assert len(injected_messages) == 1
    injected = injected_messages[0]
    assert getattr(injected, "role", None) == "user"
    assert injected.metadata.get("is_meta") is True
    assert injected.metadata.get("source") == "skill_tool"
    assert "## 当前 Runtime Skill Context" in injected.content
    assert '<skill-runtime-entry name="crypto_skill"' in injected.content
    assert "## 密码学能力" in injected.content
    assert "hash_calculator" in injected.content


def test_crypto_skill_is_fully_unloaded_after_ephemeral_cleanup(registry):
    """
    skill_tool 调用的 on-demand skill 只应在当前轮临时生效。
    清理本轮临时状态后，skill 本体和正文上下文都应被卸载。
    """
    agent = _make_agent()
    tool = SkillTool(registry, agent.skill_manager, set())

    result = tool.run({"skill_name": "crypto_skill"})
    assert "已注入 Skill `crypto_skill`" in result.to_display_string()
    assert agent.skill_manager.has_skill("crypto_skill")
    assert agent.skill_manager.is_active("crypto_skill")
    assert agent.skill_manager.has_runtime_skill_context()

    agent._clear_ephemeral_skill_state()

    assert not agent.skill_manager.has_skill("crypto_skill")
    assert not agent.skill_manager.has_runtime_skill_context()


def test_crypto_skill_becomes_system_prompt_only_when_forced_resident(registry):
    """
    如果你明确把 on-demand skill 改成 resident，
    那么它的正文才会进入 system prompt。
    """
    agent = _make_agent()

    sk = registry.create("crypto_skill")
    sk.config.exposure_mode = "resident"
    agent.skill_manager.register(sk)

    prompt = agent.get_enhanced_prompt()
    assert "## 技能与扩展能力" in prompt
    assert "## 密码学能力" in prompt
    assert "hash_calculator" in prompt
