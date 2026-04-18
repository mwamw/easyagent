import sys
import os
import time
import uuid
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, "/home/wxd/LLM/EasyAgent")

from skill.registry import SkillRegistry
from core import enable_logging
enable_logging()
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from context import ContextManager,LLMHistoryCompactor
llm2= EasyLLM(provider="openai",base_url="http://127.0.0.1:5124/v1",api_key="122",model="qwen3.5-9b")

skill_manage=SkillRegistry()
skill_manage.discover_from_directory("/home/wxd/LLM/EasyAgent/test/real_skills/")
crypto_skill=skill_manage.create('crypto_skill')

agent_context = BasicAgent(name="assistant", llm=llm2,reasoning={"effort":"high"} ,verbose_thinking=True)    
agent_context.with_skill(crypto_skill)
builder=ContextManager(max_tokens=4000)
builder.set_history_compactor(LLMHistoryCompactor(llm2,recent_turns=2))
agent_context.with_context(builder)

while 1:
    agent_context.invoke("帮我计算“hello from my home”的sha256值")

    print(agent_context.get_context_usage())