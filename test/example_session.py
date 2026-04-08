import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Optional
from core.llm import EasyLLM
from agent import BasicAgent
from dotenv import load_dotenv
load_dotenv()

llm = EasyLLM()
agent = BasicAgent(name="session_example", llm=llm)

response = agent.invoke("我是罩得住，你好，请介绍一下你自己")
print(response)

response = agent.invoke("你还记得我的名字吗？")
print(response)


print(agent.get_history())
print(agent._build_start_messages("你认为我是一个怎么样的人"))
agent.save_session("session_1001")

agent2 = BasicAgent.load_session(session_id="session_1001", llm=llm)
response = agent2.invoke("你还记得我的名字吗？")
print(response)