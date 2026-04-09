import asyncio
from core.llm import EasyLLM
from Tool.builtin import CalculatorTool
from skill.builtin.calculator_skill import CalculatorSkill

from dotenv import load_dotenv
load_dotenv("example/.env")

from pydantic import BaseModel, Field
from Tool import Tool
class TranslateParams(BaseModel):
    text: str = Field(description="要翻译的文本")
    target_lang: str = Field(default="en", description="目标语言")

class TranslateTool(Tool):
    def __init__(self):
        super().__init__("translate_tool", "将文本翻译为目标语言", TranslateParams)
    def run(self, parameters: dict) -> str:
        return f"Translated: {parameters['text']}"

async def test():
    llm = EasyLLM(provider="openai_responses")
    tools = [CalculatorTool().get_openai_tool_schema(), TranslateTool().get_openai_tool_schema()]
    
    messages = [{"role": "user", "content": "使用工具翻译文本"}]
    
    print("Testing stream...")
    try:
        async for chunk in llm._provider.async_stream_with_tools(messages, tools, reasoning={"effort": "medium"}):
            print("Got chunk!")
            break
        print("Stream works!")
    except Exception as e:
        print("Stream failed:", repr(e))

asyncio.run(test())
