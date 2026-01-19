"""测试 Provider 重构是否成功"""
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import EasyLLM
import core
core.enable_logging()

if __name__ == "__main__":
    # 测试 Provider 创建
    print("=" * 50)
    print("测试 Provider 重构")
    print("=" * 50)
    
    llm = EasyLLM(model='gemini-2.5-pro')
    print(f'Provider: {llm.provide}')
    print(f'Model: {llm.model}')
    print(f'Provider class: {type(llm.provider).__name__}')
    
    # 测试 format_tool_result
    print("\n测试 format_tool_result:")
    result = llm.format_tool_result("搜索结果", "call_123", "search")
    print(f"格式化结果: {result}")
    
    # 测试简单调用
    print("\n测试简单调用:")
    response = llm.invoke([{'role': 'user', 'content': '你好，请用一句话回复'}])
    print(f'Response: {response}')
    
    print("\n✅ Provider 重构测试通过!")
