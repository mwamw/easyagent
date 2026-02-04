import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'memory', 'V2'))

from memory.V2.EpisodicMemory import Episode
from memory.V2.Store.SQLiteDocumentStore import SQLiteDocumentStore
from memory.V2.BaseMemory import MemoryType
if __name__ == "__main__":
    db_path:str="./db/test.db"
    os.makedirs(os.path.dirname(db_path),exist_ok=True)
    store=SQLiteDocumentStore(db_path)
    store.add_memory("2","1","hello","episodic",1,1)
    store.add_memory("3","1","hello","episodic",1,1)
    store.add_memory("4 ","2","hello","episodic",1,1)
    store.add_memory("5","2","hello","episodic",1,1)
    store.add_memory("6","1","hello","episodic",1,1)
    store.add_memory("7","2","hello","episodic",1,1)
    store.add_memory("8","1","hello","episodic",1,1)
    store.add_memory("9","1","hello","episodic",1,1)
    store.add_memory("10","4","hello","episodic",1,1)
    store.add_memory("11","1","hello","episodic",1,1)
    store.add_memory("12","4","hello","episodic",1,1)
    store.add_memory("13","ssss","hello","episodic",1,1)
    print(store.get_memory("1"))
    all_memories=store.search_memory(user_id="ssss",memory_type=MemoryType.EPISODIC,limit=10)
    for memory in all_memories:
        print(memory)