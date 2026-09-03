from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from agent import BasicAgent
from agent.components.prompt_composer import PromptBuildContext, SystemPromptComposer
from core.llm import EasyLLM
from prompt import PromptBlock, SystemPromptTemplate, build_system_prompt
from Tool import Tool, ToolRegistry


class StubLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"


class DummyParams(BaseModel):
    text: str = Field(description="text")


class DummyTool(Tool):
    def __init__(self, name: str = "dummy_tool", *, guidance: str = "", prompt: str = ""):
        super().__init__(name, "dummy tool", DummyParams, guidance=guidance, prompt=prompt)

    def run(self, parameters: dict) -> str:
        return str(parameters.get("text") or "")


class ProductPromptComposer(SystemPromptComposer):
    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return [
            PromptBlock("identity", f"You are {context.agent_name}.", order=0),
            PromptBlock(
                "workspace",
                lambda current: f"Current agent: {current.agent_name}",
                placement="system_reminder",
                order=10,
            ),
        ]


def test_system_prompt_template_orders_and_filters_blocks():
    blocks = [
        PromptBlock("late", "LATE", order=20),
        PromptBlock("reminder", "REMINDER", placement="system_reminder", order=5),
        PromptBlock("early", "EARLY", order=10),
    ]
    template = SystemPromptTemplate(blocks)

    assert template.render_system() == "EARLY\n\nLATE"
    assert template.render_system_reminders() == "REMINDER"
    assert build_system_prompt(blocks, placement="system") == "EARLY\n\nLATE"


def test_prompt_block_rejects_unknown_placement():
    with pytest.raises(ValueError):
        PromptBlock("invalid", "content", placement="dynamic")


def test_basic_agent_default_prompt_is_chunked():
    registry = ToolRegistry()
    registry.register_tool(DummyTool("lookup"))
    agent = BasicAgent("assistant", StubLLM(), system_prompt="Keep it concise.").with_tool(registry)

    blocks = agent.get_system_prompt_blocks()
    assert [block.name for block in blocks[:4]] == [
        "identity",
        "visibility",
        "task_execution",
        "safety",
    ]
    assert blocks[0].content == "Keep it concise."
    assert "工具可用性始终以当前请求实际提供的 tools 集合为准" in agent.get_enhanced_prompt()


def test_skill_listing_is_separate_from_system_prompt(tmp_path: Path):
    skill_dir = tmp_path / "review"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code safely\n---\nPRIVATE BODY",
        encoding="utf-8",
    )
    agent = BasicAgent("assistant", StubLLM()).with_skill(skill_dir)
    template = agent.get_system_prompt_template()

    assert "PRIVATE BODY" not in template.render_system()
    assert "PRIVATE BODY" not in template.render_system_reminders()
    assert "`review`: Review code safely" in template.render_system_reminders()


def test_custom_composer_only_requires_build():
    agent = BasicAgent("product-agent", StubLLM()).with_prompt(ProductPromptComposer())
    blocks = agent.get_system_prompt_blocks()

    assert blocks[0].content == "You are product-agent."
    assert "Current agent: product-agent" in agent.get_system_prompt_template().render_system_reminders()
    assert "Current agent: product-agent" not in agent.get_enhanced_prompt()


def test_composer_can_replace_all_defaults():
    composer = SystemPromptComposer(
        [PromptBlock("identity", "Only this prompt")],
        include_defaults=False,
    )
    agent = BasicAgent("minimal", StubLLM()).with_prompt(composer)

    assert agent.get_enhanced_prompt() == "Only this prompt"
    assert [block.name for block in agent.get_system_prompt_blocks()] == ["identity"]


def test_extension_prompt_blocks_can_be_registered():
    composer = SystemPromptComposer().add_block(
        PromptBlock("product_policy", "## Product Policy\nUse the product contract.")
    )
    agent = BasicAgent("assistant", StubLLM()).with_prompt(composer)
    assert "## Product Policy" in agent.get_enhanced_prompt()
