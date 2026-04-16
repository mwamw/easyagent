import sys
import os
import time
import uuid
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup context
from dotenv import load_dotenv
load_dotenv()
from core import enable_logging
enable_logging()
# 将项目根目录加入模块路径
from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM
from skill.registry import SkillRegistry
from skill.builtin.calculator_skill import CalculatorSkill
from skill.yaml_loader import YAMLSkillLoader, MarkdownSkillLoader
from skill.folder_loader import FolderSkillLoader
from skill import MetaSkill
from pydantic import BaseModel,Field
from Tool import Tool
from skill import BaseSkill
from skill import SkillConfig
class TranslateParams(BaseModel):
    text: str = Field(description="要翻译的文本")
    target_lang: str = Field(default="en", description="目标语言")

class TranslateTool(Tool):
    def __init__(self):
        super().__init__("translate_tool", "将文本翻译为目标语言", TranslateParams)

    def run(self, parameters: dict) -> str:
        # 实际翻译逻辑
        return f"Translated: {parameters['text']}"

# 2. 定义 Skill
class TranslateSkill(BaseSkill):
    def __init__(self):
        config = SkillConfig(
            name="translate",
            description="多语言翻译技能",
            version="1.0.0",
            tags=["translate", "language", "i18n"],
            priority=5,
        )
        super().__init__(config)

    def get_tools(self) -> list:
        return [TranslateTool()]

    def get_prompt(self) -> str:
        return """## 翻译能力
你具备多语言翻译能力。当用户要求翻译时，请使用 translate_tool 工具。
- 支持中英日韩等多种语言
- 可以自动识别源语言
"""

def test_invoke_without_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high","summary":"auto"},verbose_thinking=True)
    result=agent.invoke("你好，请介绍一下你自己")
    print(result)

async def test_ainvoke_without_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high","summary":"auto"},verbose_thinking=True)
    result=await agent.ainvoke("你好，请介绍一下你自己")
    print(result)

def test_stream_without_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")
    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high"},verbose_thinking=True)

    agent.stream_invoke("你好，请介绍一下你自己")

async def test_astream_without_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high"},verbose_thinking=True)

    await agent.astream_invoke("你好，请介绍一下你自己")

def test_invoke_with_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high"},verbose_thinking=True)
    agent.with_skill(CalculatorSkill())
    agent.with_skill(TranslateSkill())
    result=agent.invoke(f"使用工具翻译下面的文字到英语并判断这个工具正确吗:\n你是谁，在哪里 \n 并帮我计算3^22")
    print(result)

async def test_ainvoke_with_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high"},verbose_thinking=True)
    agent.with_skill(CalculatorSkill())
    agent.with_skill(TranslateSkill())
    result=await agent.ainvoke(f"使用工具翻译下面的文字到英语并判断这个工具正确吗:\n你是谁，在哪里 \n 并帮我计算3^22")
    print(result)

def test_stream_with_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high"},verbose_thinking=True)
    agent.with_skill(CalculatorSkill())
    agent.with_skill(TranslateSkill())
    agent.stream_invoke(f"使用工具翻译下面的文字到英语并判断这个工具正确吗:\n你是谁，在哪里 \n 并帮我计算3^22")

async def test_astream_with_tool():
    llm= EasyLLM(provider="openai_responses",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

    agent=BasicAgent(name="test_skill", llm=llm,reasoning={"effort":"high"},verbose_thinking=True)
    agent.with_skill(CalculatorSkill())
    agent.with_skill(TranslateSkill())
    await agent.astream_invoke(f"使用工具翻译下面的文字到英语并判断这个工具正确吗:\n你是谁，在哪里 \n 并帮我计算3^22")

import asyncio
# test_invoke_without_tool()
# test_ainvoke_without_tool()
# test_stream_without_tool()
# test_astream_without_tool()
test_invoke_with_tool()
# test_ainvoke_with_tool()
# test_stream_with_tool()
# asyncio.run(test_astream_with_tool())

