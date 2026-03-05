import sys
import os  
import json
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'memory', 'V2'))
from core.llm import EasyLLM
from memory.V2.Extractor.Extractor import Extractor

llm = EasyLLM()

# 测试复杂文本（更能体现验证 Agent 的价值）
content = '张三，25岁，是一名软件工程师，在华为工作。他负责管理一个叫做"智慧城市"的项目。李四是张三的同事，也参与了这个项目。'

print("=== 带验证的提取 ===")
extractor = Extractor(llm, enable_verification=True)
result = extractor.extract(content)
for e in result["entities"]:
    print(f"  实体: {e.name} ({e.entity_type}) - {e.description}")
for r in result["relations"]:
    print(f"  关系: {r.from_entity} -[{r.relation_type}]-> {r.to_entity} (strength: {r.strength})")

print("\n=== GraphStore 格式 ===")
graph_data = extractor.extract_for_graph(content)
for entity in graph_data["entities"]:
    print(f"  实体: {entity.to_dict()}")
for relation in graph_data["relations"]:
    print(f"  关系: {relation.to_dict()}")
