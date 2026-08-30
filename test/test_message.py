import sys
import os  
import json
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.Message import Message,UserMessage,AssistantMessage,SystemMessage,ToolMessage

messages:list[dict[str,str]]=[]

messages.append(UserMessage(content="hello").to_dict())

print(messages)