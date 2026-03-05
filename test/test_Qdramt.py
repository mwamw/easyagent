import sys
import os
import numpy as np
import uuid
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'memory', 'V2'))
from memory.V2.Store.QdrantVectorStore import QdrantVectorStore
if __name__ == "__main__":
    print("----------------连接验证--------------------")
    qdrant_vector_store = QdrantVectorStore(
        way="memory",
        collection_name="test",
        vector_size=384,
    )
    print("----------------添加数据--------------------")
    qdrant_vector_store.add_vectors(
        vectors=[np.random.rand(384).tolist(),np.random.rand(384).tolist()],
        metadata=[{"name": "test"},{"name": "test"}],
        ids=[str(uuid.uuid4()),str(uuid.uuid4())]
    )
    print("----------------获取所有向量--------------------")
    all_vectors = qdrant_vector_store.get_all_vectors()
    print(all_vectors)
    
    print("----------------查询相似向量--------------------")
    query_embedding = np.random.rand(384).tolist()
    search_result = qdrant_vector_store.search_similar(query_embedding, {}, 10)
    print(search_result[0]["similarity"])

    print("----------------删除向量--------------------")
    qdrant_vector_store.remove_vectors([all_vectors[0]["memory_id"]])
    all_vectors = qdrant_vector_store.get_all_vectors()
    print(all_vectors)

    