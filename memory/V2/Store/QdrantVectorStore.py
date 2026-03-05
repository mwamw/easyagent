from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance,PointStruct,PointIdsList,ExtendedPointId,Filter,FieldCondition,MatchValue
from typing import Optional,Any,cast,List
from BaseMemory import MemoryType
try:
    from .VectorStore import VectorStore
except ImportError:
    from Store.VectorStore import VectorStore
from logging import getLogger
logger = getLogger(__name__)
class QdrantVectorStore(VectorStore):
    def __init__(self,way:str,collection_name:str,host:Optional[str]=None,port:Optional[int]=None,api_key:Optional[str]=None,vector_size:Optional[int]=None,distance:Optional[Distance]=Distance.COSINE,timeout:Optional[int]=10):
        assert way in ["memory","cloud","local"]
        if way == "memory":
            self.client = QdrantClient(
                ":memory:",
                timeout=timeout
            )
            logger.info("✅成功连接到Qdrant内存")
        elif way == "cloud":
            if not host or not api_key:
                raise ValueError("Qdrant云服务需要host和api_key")
            self.client = QdrantClient(
                url=host,
                api_key=api_key,
                timeout=timeout
            )
            logger.info(f"✅ 成功连接到Qdrant云服务: {host}")
        else:
            if not host or not port:
                raise ValueError("Qdrant本地服务需要host和port")
            self.client = QdrantClient(
                host=host,
                port=port,
                timeout=timeout
            )
            logger.info(f"✅ 成功连接到Qdrant服务: {host}:{port}")

        self.collection_name = collection_name
        self.vector_size = vector_size if vector_size else 384
        self.distance = distance if distance else Distance.COSINE
        self.timeout = timeout if timeout else 10
        self._sure_collection()

    def _sure_collection(self):
        if self.collection_name not in self.client.get_collections().collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            logger.info(f"✅ 成功创建集合: {self.collection_name}")
        else:
            logger.info(f"✅ 集合已存在: {self.collection_name}")

    def add_vectors(self, vectors: list[list[float]], metadata: list[dict[str, Any]], ids: list[str]) -> str:
        points=[]
        assert len(vectors)==len(metadata)==len(ids)
        for i in range(len(vectors)):
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=metadata[i]
                )
            )
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"✅ 成功添加{len(vectors)}个向量")
        except Exception as e:
            logger.error(f"❌ 添加向量失败: {e}")
            return "fail"
        return "success"
    
    def remove_vectors(self,ids:list[str]):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=cast(List[ExtendedPointId], ids))
            )
            logger.info(f"✅ 成功删除{len(ids)}个向量")
        except Exception as e:
            logger.error(f"❌ 删除向量失败: {e}")
            return 0
        return len(ids)

    def clear_type_memory(self,memory_type:MemoryType):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="memory_type", match=MatchValue(value=memory_type.value))]
                )
            )
            logger.info(f"✅ 成功删除{memory_type.value}类型的向量")
        except Exception as e:
            logger.error(f"❌ 删除向量失败: {e}")
            return 0
        return 1

    def get_collection_stats(self)->dict[str,Any]:
        try:
            stats = self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"✅ 成功获取集合统计信息")
        except Exception as e:
            logger.error(f"❌ 获取集合统计信息失败: {e}")
            return {}
        return dict(stats)

    def search_similar(self,query_embedding:list[float],where:dict[str,Any],limit:int)->list[dict[str,Any]]:
        try:
            # 构建查询过滤器
            query_filter = None
            if where:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in where.items()
                ]
                query_filter = Filter(must=conditions) #type: ignore
            
            search_result = self.client.query_points( 
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=True
            ).points
            logger.info(f"✅ 成功搜索到{len(search_result)}个向量")
        except Exception as e:
            logger.error(f"❌ 搜索向量失败: {e}")
            return []
        dict_result=[]
        for point in search_result:
            dict_result.append({
                "memory_id":point.id,
                "similarity":point.score,
                "vector":point.vector,
                "metadata":point.payload
            })
        return dict_result
    
    def get_all_vectors(self,with_vector=False):
        try:
            all_vectors = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=with_vector
            )
            logger.info(f"✅ 成功获取所有向量")
        except Exception as e:
            logger.error(f"❌ 获取所有向量失败: {e}")
            return []
        dict_result=[]
        for point in all_vectors[0]:
            dict_result.append({
                "memory_id":point.id,
                "vector":point.vector,
                "metadata":point.payload
            })
        return dict_result
    
    def delete_collection(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"✅ 成功删除集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ 删除集合失败: {e}")
            return 0
        return 1