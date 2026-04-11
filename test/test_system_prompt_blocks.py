import os
import sys
import unittest
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pydantic import BaseModel, Field

from agent.BasicAgent import BasicAgent
from agent.ReactAgent import ReactAgent
from core.llm import EasyLLM
from memory import MemoryConfig, MemoryManage, WorkingMemory
from prompt import PromptBlock, SystemPromptTemplate, build_system_prompt
from skill.base import BaseSkill, SkillConfig
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry


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


class DummyParams(BaseModel):
    text: str = Field(description="text")


class DummyTool(Tool):
    def __init__(self, name: str = "dummy_tool"):
        super().__init__(name, "dummy tool", DummyParams)

    def run(self, parameters: dict):
        return parameters.get("text", "")


class ExtraSkill(BaseSkill):
    def __init__(self, name: str = "extra_skill", prompt: str = "## Extra Skill\nUse it when helpful."):
        super().__init__(SkillConfig(name=name, priority=5))
        self._prompt = prompt
        self._tools = [DummyTool(f"{name}_tool")]

    def get_tools(self) -> list[Tool]:
        return self._tools

    def get_prompt(self) -> str:
        return self._prompt


class TestSystemPromptBlocks(unittest.TestCase):
    def test_system_prompt_template_keeps_block_order(self):
        blocks = [
            PromptBlock(name="late", content="LATE", order=20),
            PromptBlock(name="early", content="EARLY", order=10),
        ]

        template = SystemPromptTemplate(blocks)

        self.assertEqual([block.name for block in template.get_blocks()], ["early", "late"])
        self.assertEqual(template.render(), "EARLY\n\nLATE")
        self.assertEqual(build_system_prompt(blocks), "EARLY\n\nLATE")

    def test_basic_agent_returns_chunked_system_prompt(self):
        llm = SpyEasyLLM()
        registry = ToolRegistry()
        registry.register_tool(DummyTool("lookup"))

        agent = BasicAgent(
            name="assistant",
            llm=llm,
            tool_registry=registry,
            enable_tool=True,
            system_prompt="请保持回答简洁。",
        )

        blocks = agent.get_system_prompt_blocks()
        block_names = [block.name for block in blocks]

        self.assertEqual(block_names[:4], ["identity", "visibility", "task_execution", "safety"])
        self.assertIn("custom_instructions", block_names)
        self.assertNotIn("tool_inventory", block_names)
        prompt = agent.get_enhanced_prompt()
        self.assertIn("工具调用之外输出的所有文本都会直接展示给用户", prompt)
        self.assertIn("多个互不依赖的工具调用应并行执行", prompt)
        self.assertNotIn("参数 Schema", prompt)

    def test_memory_block_is_not_duplicated_inside_skills_block(self):
        llm = SpyEasyLLM()
        agent = BasicAgent(name="assistant", llm=llm)

        memory_manage = MemoryManage(
            MemoryConfig(),
            working_memory=WorkingMemory(MemoryConfig()),
            enable_episodic=False,
            enable_semantic=False,
        )
        memory_manage.add_memory(
            content="预算上限 5000 元",
            memory_type="working",
            importance=1.0,
            metadata={"source": "test"},
        )

        agent.with_memory(memory_manage)
        prompt = agent.get_enhanced_prompt()
        block_names = [block.name for block in agent.get_system_prompt_blocks()]

        self.assertIn("memory", block_names)
        self.assertNotIn("skills", block_names)
        self.assertEqual(prompt.count("## 记忆系统"), 1)
        self.assertIn("### 1. 记忆的目的", prompt)
        self.assertIn("### 2. 何时访问记忆", prompt)
        self.assertIn("### 7. 当前 Working Memory 展示", prompt)
        self.assertIn("`working` (Working Memory): 已启用", prompt)
        self.assertIn("`semantic` (Semantic Memory): 未启用", prompt)
        self.assertNotIn("dict_keys", prompt)
        self.assertIn("预算上限 5000 元", prompt)

    def test_non_memory_skills_still_render_after_memory_block(self):
        llm = SpyEasyLLM()
        agent = BasicAgent(name="assistant", llm=llm)

        memory_manage = MemoryManage(
            MemoryConfig(),
            working_memory=WorkingMemory(MemoryConfig()),
            enable_episodic=False,
            enable_semantic=False,
        )
        agent.with_memory(memory_manage)
        agent.with_skill(ExtraSkill())

        prompt = agent.get_enhanced_prompt()
        blocks = agent.get_system_prompt_blocks()
        block_names = [block.name for block in blocks]

        self.assertIn("memory", block_names)
        self.assertIn("skills", block_names)
        self.assertEqual(prompt.count("## 记忆系统"), 1)
        self.assertIn("### 4. 哪些内容禁止写入", prompt)
        self.assertIn("### 6. 基于记忆回答前如何验证", prompt)
        self.assertIn("## 技能与扩展能力", prompt)
        self.assertIn("## Extra Skill", prompt)

    def test_react_agent_exposes_react_specific_blocks(self):
        llm = SpyEasyLLM()
        registry = ToolRegistry()
        registry.register_tool(DummyTool("lookup"))

        agent = ReactAgent(
            name="react",
            llm=llm,
            tool_registry=registry,
            enable_tool=True,
        )

        blocks = agent.get_system_prompt_blocks()
        self.assertEqual(blocks[0].name, "identity")
        self.assertEqual(blocks[1].name, "react_workflow")
        prompt = agent.get_enhanced_prompt()
        self.assertIn("Final Answer", prompt)
        self.assertIn("## 可用工具概览", prompt)
        self.assertIn("- lookup: dummy tool", prompt)
        self.assertNotIn("参数 Schema", prompt)

    def test_extension_prompt_blocks_can_be_registered(self):
        llm = SpyEasyLLM()
        agent = BasicAgent(name="assistant", llm=llm)
        agent.with_prompt_block(
            PromptBlock(
                name="claude_optional",
                content="## Claude Optional\n这是预留给 Claude Code 专属能力的扩展块。",
            )
        )

        prompt = agent.get_enhanced_prompt()
        self.assertIn("## Claude Optional", prompt)
