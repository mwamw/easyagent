# 必须先配置路径，再导入自定义模块
import sys
import os  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import EasyLLM
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    llm:EasyLLM=EasyLLM(model="Qwen3",provide="custom")
    print(llm.model,llm.provide,llm.resovle_api_key,llm.resovle_base_url)
    print(llm.invoke([{"role":"user","content":"你是什么模型"}]))