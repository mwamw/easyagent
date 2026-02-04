import datetime
import datetime
from heapq import heappush
from typing_extensions import override
from BaseMemory import BaseMemory,MemoryConfig,MemoryItem,ForgetType
import logging
import heapq
from typing import Optional,Any
from abc import abstractmethod
from Embedding.BaseEmbeddingModel import BaseEmbeddingModel
import numpy as np
logger = logging.getLogger(__name__)
"""
工作记忆：
    - 保存在内存当中，只记录单次会话
    - 可以提供总结功能，需提供EasyLLM
    - 提供多类别的清理机制，自动清理机制
"""
class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig,embedding_model:Optional[BaseEmbeddingModel]=None):
        super().__init__(config)

        #工作记忆特有配置
        self.embedding_model=embedding_model
        self.max_capacity=self.config.max_capacity
        self.max_token=self.config.max_working_token

        self.time_cap_min=getattr(self.config,"working_memory_ttl",120)

        self.current_tokens=0
        self.size=0
        self.session_time=datetime.datetime.now()
        self.memory_list:list=[]

        self.memory_heap:list=[]  # (priority, timestamp, memory_item)

    @override
    def add_memory(self, item: MemoryItem) -> str:
        #清理过期记忆
        self._clean_expired()
        if self.size >=self.max_capacity or self.current_tokens >=self.max_token:
            # 清理最低优先级的记忆
            self._clean_lowest_priority()
        priority:float=self._calculate_priority(item)
        self.memory_list.append(item)
        self.size+=1
        self.current_tokens+=len(item.content.split())
        heapq.heappush(self.memory_heap,(-priority,item.timestamp,item))
        return item.id

    def _clean_expired(self):
        
        max_timestamp=datetime.datetime.now()-datetime.timedelta(minutes=self.time_cap_min)
        new_memory_list:list=[]
        for memory in self.memory_list:
            if memory.timestamp < max_timestamp:
                continue
            new_memory_list.append(memory)
        self.memory_list=new_memory_list
        self.size=len(self.memory_list)
        self.current_tokens=sum([len(memory.content.split()) for memory in self.memory_list])
        
        #重建堆
        self._rebuild_heap()

    def _rebuild_heap(self):
        self.memory_heap=[]
        for memory in self.memory_list:
            priority=self._calculate_priority(memory)
            heapq.heappush(self.memory_heap,(-priority,memory.timestamp,memory))
    def _clean_lowest_priority(self):
        while self.size > self.max_capacity or self.current_tokens > self.max_token:
            _,_,memory=heapq.heappop(self.memory_heap)
            if memory:
                self.memory_list.remove(memory)
                self.size-=1
                self.current_tokens-=len(memory.content.split())

    def _calculate_priority(self, memory_item: MemoryItem) -> float:
        """计算记忆的优先级"""
        # 基于时间衰减
        time_decay = self._calculate_time_decay(memory_item.timestamp)
        
        # 基于重要性
        importance_weight = memory_item.importance
        
        # 计算最终优先级
        priority = time_decay * importance_weight
        return priority

    def _calculate_time_decay(self,timestamp:datetime.datetime)->float:
        time_diff=datetime.datetime.now()-timestamp
        hours_diff=time_diff.total_seconds()/3600

        dacay_factor=self.config.decay_factor ** (hours_diff/6)

        return max(0.1,dacay_factor)

    @override
    def remove_memory(self,memory_id:str)->bool:
        for memory in self.memory_list:
            if memory.id==memory_id:
                self.memory_list.remove(memory)
                self.size-=1
                self.current_tokens-=len(memory.content.split())
                self._rebuild_heap()
                return True
        return False
    
    @override
    def update_memory(self,id:str,content:str,importance:Optional[float]=None,metadata:Optional[dict[str,Any]]=None):
        for memory in self.memory_list:
            if memory.id ==id:
                new_memory=memory.model_copy()
                new_memory.content=content
                new_memory.timestamp=datetime.datetime.now()
                if importance:
                    new_memory.importance=importance
                if metadata:
                    new_memory.metadata=metadata
                self.current_tokens-=len(memory.content.split())
                self.current_tokens+=len(new_memory.content.split())
                self.remove_memory(id)
                self.add_memory(new_memory)
                self._rebuild_heap()
                return True
        return False

    @override
    def find_memory(self,id:str)->bool:
        for memory in self.memory_list:
            if memory.id ==id:
                return True
        return False

    @override
    def clear_memory(self):
        self.memory_list=[]
        self.size=0
        self.current_tokens=0
        self.memory_heap=[]

    @override
    def get_stats(self) -> dict[str, Any]:
        """获取工作记忆统计信息"""
        # 过期清理（惰性）
        self._clean_expired()
        
        # 工作记忆中的记忆都是活跃的（已遗忘的记忆会被直接删除）
        active_memories = self.memory_list
        
        return {
            "count": len(active_memories),  # 活跃记忆数量
            "forgotten_count": 0,  # 工作记忆中已遗忘的记忆会被直接删除
            "total_count": len(self.memory_list),  # 总记忆数量
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_token,
            "max_age_minutes": self.time_cap_min,
            "session_duration_minutes": (datetime.datetime.now() - self.session_time).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_token if self.max_token > 0 else 0.0,
            "memory_type": "working"
        }

    def get_recent_memories(self,limit:int=5)->list[MemoryItem]:
        """
        获取最近的记忆
        """
        return self.memory_list[-limit:]

    def get_important_memories(self,limit:int=5)->list[MemoryItem]:
        """
        获取最重要记忆
        """
        return sorted(self.memory_list,key=lambda x:x.importance,reverse=True)[:limit]

    def get_all_memories(self)->list[MemoryItem]:
        """
        获取所有记忆
        """
        return self.memory_list.copy()
    
    def search_memory(self, query: str, limit: int = 5,user_id:Optional[str]=None, **kwargs):
        """
        根据query查询记忆,支持向量检索以及关键字检索
        """
        active_memories=[memory for memory in self.memory_list if memory.metadata.get("forgotten",False)==False]
        if user_id:
            active_memories=[memory for memory in active_memories if memory.user_id==user_id]
        
        if not active_memories:
            return []
        #计算向量相似度
        vector_similarities=[]
        try:
            memories_text=[memory.content for memory in active_memories]
            if self.embedding_model:

                query_embedding=self.embedding_model.embed([query])[0]
                memories_embedding=self.embedding_model.embed(memories_text)
                vector_similarities=self.cosine_similarity(query_embedding,memories_embedding)
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
                text=[query]+memories_text
                tfidf_matrix=vectorizer.fit_transform(text)
                query_tfidf=tfidf_matrix[0]
                memories_tfidf=tfidf_matrix[1:]
                vector_similarities=cosine_similarity(query_tfidf,memories_tfidf).flatten().tolist()
                print("向量相似度",vector_similarities)
        except Exception as e:
            print(e)
            vector_similarities=[0.0]*len(active_memories)
        
        # 计算关键字相似度
        keyword_similarities=[]
        query_keywords=set(query.lower().split())
        for memory in active_memories:
            memory_keywords=set(memory.content.lower().split())
            keyword_similarities.append(len(query_keywords.intersection(memory_keywords))/len(query_keywords.union(memory_keywords)))
        
        print("关键字相似度",keyword_similarities)
        # 综合相似度
        combined_similarities=np.array(vector_similarities)*0.7+np.array(keyword_similarities)*0.3
        print("综合相似度",combined_similarities)
        # 时间衰减
        time_decay=np.array([self._calculate_time_decay(memory.timestamp) for memory in active_memories])
        print("时间衰减",time_decay)
        # 记忆重要性加权
        final_scores=combined_similarities*time_decay*np.array([memory.importance*0.2+0.8 for memory in active_memories])
        print("最终得分",final_scores)
        
        # 排序
        sorted_indices=np.argsort(final_scores)[::-1]
        
        # 返回前limit个记忆
        return [active_memories[i] for i in sorted_indices[:limit]]
        

    def cosine_similarity(self,query_embedding:list[float],memories_embedding:list[list[float]])->list[float]:
        """
        计算余弦相似度
        """
        query_array=np.array(query_embedding)
        memories_array=np.array(memories_embedding)
        similarities=np.dot(query_array,memories_array.T)/(np.linalg.norm(query_array)*np.linalg.norm(memories_array,axis=1))
        return similarities.tolist()


    def forget(self,base_on:ForgetType,threshold:float=0.1,max_age_days:int=1):
        """
        根据base_on和threshold以及max_age_days来遗忘记忆
        """
        now_size=self.size
        #删除过期记忆
        if base_on==ForgetType.TIME:
            self._clean_expired()
        #删除低重要性记忆
        if base_on==ForgetType.IMPORTANCE:
            self._clean_importance(threshold)

        if base_on==ForgetType.CAPACITY:
            self._clean_lowest_priority()
        
        return now_size-self.size

    def _clean_importance(self,threshold:float):
        """
        删除低重要性记忆
        """
        self.memory_list=[memory for memory in self.memory_list if memory.importance>=threshold]
        self.size=len(self.memory_list)
        self.current_tokens=sum([memory.token_count for memory in self.memory_list])
        self._rebuild_heap()

if __name__ == "__main__":
    working_memory=WorkingMemory(MemoryConfig())

    #测试添加
    print("----------------------------测试添加 ----------------------------------")
    memory1:MemoryItem=MemoryItem(
        id="1",
        content="hello,i am wxd who is a student of ustc, i am 20 years old, i am a boy",
        type="working",
        user_id="1",
        timestamp=datetime.datetime.now(),
        importance=1,
        metadata={}
    )
    working_memory.add_memory(memory1)
    memory2:MemoryItem=MemoryItem(
        id="2",
        content="i am focus on computer science",
        type="working",
        user_id="1",
        timestamp=datetime.datetime.now(),
        importance=1,
        metadata={}
    )
    working_memory.add_memory(memory2)
    print(working_memory.get_all_memories())

    #测试查询
    print("----------------------------测试查询 ----------------------------------")
    print(working_memory.search_memory("what do i like",limit=1))


    print("----------------------------测试更新 ----------------------------------")
    working_memory.update_memory("1","hello,i am wxd who is a student of ustc, i am 20 years old, i am a boy",importance=0.5)
    print(working_memory.get_all_memories())
    print(working_memory.memory_heap)
    