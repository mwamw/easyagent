from datetime import datetime
import uuid
from numpy import extract
from pydantic import config
from BaseMemory import MemoryConfig, MemoryItem,BaseMemory
from WorkingMemory import WorkingMemory
from EpisodicMemory import EpisodicMemory
from SemanticMemory import SemanticMemory
from PerceptualMemory import PerceptualMemory
from Embedding.BaseEmbeddingModel import BaseEmbeddingModel
from typing import Optional
import os
from typing import Dict,Any
import logging
from BaseMemory import ForgetType
logger=logging.getLogger(__name__)
class MemoryManage:
    def __init__(
        self,
        config:MemoryConfig,
        user_id:str="default_user",
        enable_working:bool=True,
        working_memory:Optional[WorkingMemory]=None,
        enable_episodic:bool=True,
        episodic_memory:Optional[EpisodicMemory]=None,
        enable_semantic:bool=True,
        semantic_memory:Optional[SemanticMemory]=None,
        enable_perceptual:bool=False,
        perceptual_memory:Optional[PerceptualMemory]=None,
    ):
        self.config=config or MemoryConfig()
        self.user_id=user_id
        self.memory_types:Dict[str,BaseMemory]={}
        if enable_working:
            if working_memory:
                self.memory_types['working']=working_memory
            else:
                self.memory_types['working']=WorkingMemory(self.config)
        if enable_episodic:
            if episodic_memory:
                self.memory_types['episodic']=episodic_memory
            else:
                from Store import SQLiteDocumentStore,QdrantVectorStore
                from Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
                episode_document_store=SQLiteDocumentStore(os.getenv("EPISODIC_SQLITE_PATH") or "./EpisodicMemory.db")
                episodic_vector_store=QdrantVectorStore(way="local",collection_name="episodic_memory",host=os.getenv("QDRANT_HOST") or "localhost",port=int(os.getenv("QDRANT_PORT") or 6379),vector_size=int(os.getenv("QDRANT_VECTOR_SIZE") or 384))
                embedding_model=HuggingfaceEmbeddingModel(os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
                self.memory_types['episodic']=EpisodicMemory(self.config,episode_document_store,episodic_vector_store,embedding_model)
        if enable_semantic:
            if semantic_memory:
                self.memory_types['semantic']=semantic_memory
            else:
                from Store import SQLiteDocumentStore,Neo4jGraphStore
                from Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
                semantic_vector_store=QdrantVectorStore(way="local",collection_name="semantic_memory",host=os.getenv("QDRANT_HOST") or "localhost",port=int(os.getenv("QDRANT_PORT") or 6379),vector_size=int(os.getenv("QDRANT_VECTOR_SIZE") or 384))
                semantic_graph_store=Neo4jGraphStore(uri=os.getenv("NEO4J_URI") or "bolt://localhost:7687",username=os.getenv("NEO4J_USER") or "neo4j",password=os.getenv("NEO4J_PASSWORD") or "password",database=os.getenv("NEO4J_DATABASE") or "neo4j")
                embedding_model=HuggingfaceEmbeddingModel(os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
                from Extractor.Extractor import Extractor
                from core.llm import EasyLLM
                llm=EasyLLM()
                extractor=Extractor(llm)
                self.memory_types['semantic']=SemanticMemory(self.config,semantic_vector_store,semantic_graph_store,extractor,embedding_model)
        if enable_perceptual:
            if perceptual_memory:
                self.memory_types['perceptual']=perceptual_memory
            else:
                from Store import SQLiteDocumentStore,QdrantVectorStore
                from Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
                perceptual_document_store=SQLiteDocumentStore(os.getenv("PERCEPTUAL_SQLITE_PATH") or "./PerceptualMemory.db")
                perceptual_vector_store=QdrantVectorStore(way="local",collection_name="perceptual_memory",host=os.getenv("QDRANT_HOST") or "localhost",port=int(os.getenv("QDRANT_PORT") or 6379),vector_size=int(os.getenv("QDRANT_VECTOR_SIZE") or 384))
                embedding_model=HuggingfaceEmbeddingModel(os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
                self.memory_types['perceptual']=PerceptualMemory(memory_config=self.config,
                    document_store=perceptual_document_store,
                    vector_stores={"text":perceptual_vector_store},
                    embedding_model=embedding_model)
        logger.info("MemoryManage init success")
        logger.info(f"MemoryManage init success, memory types: {self.memory_types.keys()}")


    def add_memory(self,content:str,memory_type:str,importance:float,metadata:Optional[Dict[str,Any]]=None,auto_classify:bool=True)->str:
        if auto_classify:
            memory_type=memory_type or self._classify_memory_type(content,metadata)
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory type: {memory_type}")
        
        memory_item=MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )

        try:
            memory_id=self.memory_types[memory_type].add_memory(memory_item)
            logger.debug(f"Add memory to {memory_type}: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory to {memory_type}: {e}")
            raise ValueError(f"Failed to add memory to {memory_type}: {e}")



    def _classify_memory_type(self,content:str,metadata:Optional[Dict[str,Any]]=None)->str:
        if metadata and metadata.get("type"):
            return metadata["type"]
        if metadata and "raw_data" in metadata:
            return "perceptual"
        if self._is_episodic_content(content):
            return "episodic"
        if self._is_semantic_content(content):
            return "semantic"
        return "working"
        

    def _is_episodic_content(self, content: str) -> bool:
        """判断是否为情景记忆内容"""
        episodic_keywords = ["昨天", "今天", "明天", "上次", "记得", "发生", "经历"]
        return any(keyword in content for keyword in episodic_keywords)
    
    def _is_semantic_content(self, content: str) -> bool:
        """判断是否为语义记忆内容"""
        semantic_keywords = ["定义", "概念", "规则", "知识", "原理", "方法"]
        return any(keyword in content for keyword in semantic_keywords)


    def remove_memory(self,memory_id:str)->bool:
        for key,value in self.memory_types.items():
            if value.find_memory(memory_id):
                return value.remove_memory(memory_id)
        logger.warning(f"Memory {memory_id} not found")
        return False

    def search_memory(self,query:str,memory_types:Optional[list[str]]=None,limit:int=10,importance_threshold:float=0.0,time_range:Optional[tuple]=None,session_id:Optional[str]=None)->list[MemoryItem]:
        if not memory_types:
            memory_types=list(self.memory_types.keys())
        all_results:list[MemoryItem]=[]
        for memory_type in memory_types:
            if memory_type not in self.memory_types:
                continue
            try:
                results=self.memory_types[memory_type].search_memory(query,limit,user_id=self.user_id,importance_threshold=importance_threshold,time_range=time_range,session_id=session_id)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to search memory in {memory_type}: {e}")
        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results


    def update_memory(self,memory_id:str,content:str,importance:Optional[float]=None,metadata:Optional[Dict[str,Any]]=None)->bool:
        for key,value in self.memory_types.items():
            if value.find_memory(memory_id):
                return value.update_memory(memory_id,content=content,importance=importance,metadata=metadata)
        logger.warning(f"Memory {memory_id} not found")
        return False
    
    def find_memory(self,memory_id:str)->bool:
        for key,value in self.memory_types.items():
            if value.find_memory(memory_id):
                return True
        return False

    def forget_memory(self,strategy:str,threshold:float=0.1,max_age_days:int=30)->int:
        forget_count=0
        for key,value in self.memory_types.items():
            if strategy==ForgetType.CAPACITY.value:
                forget_count+=value.forget(ForgetType.CAPACITY,threshold,max_age_days)
            elif strategy==ForgetType.IMPORTANCE.value:
                forget_count+=value.forget(ForgetType.IMPORTANCE,threshold,max_age_days)
            elif strategy==ForgetType.TIME.value:
                forget_count+=value.forget(ForgetType.TIME,threshold,max_age_days)
            else:
                raise ValueError(f"Invalid forget strategy: {strategy}")
        logger.info(f"Forget {forget_count} memories")
        return forget_count
    
    def get_all_memories(self)->list[MemoryItem]:
        all_memories:list[MemoryItem]=[]
        for key,value in self.memory_types.items():
            all_memories.extend(value.get_all_memories())
        return all_memories
    
    def get_memory_stats(self) -> dict[str, Any]:
        """获取记忆统计信息"""
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor
            }
        }

        for memory_type, memory_instance in self.memory_types.items():
            type_stats = memory_instance.get_stats()
            stats["memories_by_type"][memory_type] = type_stats
            stats["total_memories"] += type_stats.get("count", 0)

        return stats
    def clear_memories(self):
        clear_count=0
        for key,value in self.memory_types.items():
            value.clear_memory()

    def merge_memories(self,source_type:str,target_type:str,importance_threshold:float=0.5):
        if source_type not in self.memory_types or target_type not in self.memory_types:
            raise ValueError(f"Invalid memory type: {source_type} or {target_type}")
        source_memory=self.memory_types[source_type]
        target_memory=self.memory_types[target_type]
        all_memories=source_memory.get_all_memories()
        merged_count=0
        merged_memories:list[MemoryItem]=[]
        for memory in all_memories:
            if memory.importance>=importance_threshold:
                merged_memories.append(memory)
        for memory in merged_memories:
            if source_memory.remove_memory(memory.id):
                target_memory.add_memory(memory)
                merged_count+=1
        logger.info(f"Merged {merged_count} memories from {source_type} to {target_type}")        
            
        return merged_count

    def sync_memories(self):
        for memory_type,memory_instance in self.memory_types.items():
            memory_instance.sync_stores()
        logger.info("Synced all memories")
        
    
    def load_memories(self):
        for memory_type,memory_instance in self.memory_types.items():
            memory_instance.load_from_store()
        logger.info("Loaded all memories")