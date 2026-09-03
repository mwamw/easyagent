# 必须先配置路径，再导入自定义模块
import sys
import os  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import EasyLLM
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    llm:EasyLLM=EasyLLM()
    print(llm.model, llm.provider_name, llm.api_key, llm.base_url)
    print(llm.invoke([{"role":"user","content":"你是什么模型"}]))
