from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel
from typing import Any, List, Optional
from enum import Enum

class ForgetType(Enum):
    TIME="time"
    IMPORTANCE="importance"
    CAPACITY="capacity"

class MemoryItem(BaseModel):
    id:str
    content:str
    type:str
    user_id:str
    timestamp:datetime
    importance:float
    metadata:dict[str,Any]

class MemoryType(Enum):
    EPISODIC="episodic"
    WORKING="working"
    SEMANTIC="semantic"
    PERCEPTUAL="perceptual"

class MemoryConfig(BaseModel):

    #基础设置
    max_capacity:int=100
    importance_threshold:float=0.1
    decay_factor:float=0.95

    #工作记忆配置
    max_working_token:int=10000

    #批量操作配置
    batch_size:int=50  # 批量 embedding 的大小

class BaseMemory(ABC):
    """记忆基类
    需要提供以下功能:
    - 添加记忆 (单条/批量/异步)
    - 删除记忆
    - 更新记忆
    - 随机生成记忆id
    - 根据查询搜索记忆 (同步/异步)
    - 查找是否存在记忆
    - 清除记忆
    - 获取记忆统计信息
    - 计算记忆重要性
    """

    def __init__(self,config:MemoryConfig):
        self.config=config
        self.memory_type=self.__class__.__name__.replace("memory","")

    # ==================== 同步方法 ====================

    def sync_stores(self) -> None:
        """同步各存储层数据 (默认空实现，子类可覆写)"""
        pass

    def load_from_store(self) -> None:
        """从存储层加载数据 (默认空实现，子类可覆写)"""
        pass

    @abstractmethod
    def add_memory(self,item:MemoryItem)->str:
        """添加单条记忆

        Args:
            item: 记忆对象

        Returns:
            记忆ID，失败返回空字符串
        """
        pass

    def add_memories_batch(self, items: List[MemoryItem]) -> List[str]:
        """批量添加记忆 (默认实现，子类可覆写以优化)

        Args:
            items: 记忆对象列表

        Returns:
            成功添加的记忆ID列表
        """
        results = []
        for item in items:
            result = self.add_memory(item)
            if result:
                results.append(result)
        return results

    @abstractmethod
    def remove_memory(self,memory_id:str) -> bool:
        pass

    @abstractmethod
    def update_memory(self,id:str,content:str,importance:Optional[float]=None,metadata:Optional[dict[str,Any]]=None) -> bool:
        pass

    def _genid(self):
        import uuid
        return str(uuid.uuid4())

    @abstractmethod
    def search_memory(self,query:str,limit:int=5,user_id:Optional[str]=None,**kwargs) -> List[MemoryItem]:
        pass

    @abstractmethod
    def find_memory(self,id:str)->bool:
        pass
    
    @abstractmethod
    def get_memory(self,ids:list[str])->list[MemoryItem]:
        pass
    @abstractmethod
    def clear_memory(self):
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典
        """
        pass
    
    @abstractmethod
    def forget(self, strategy:ForgetType, threshold: float = 0.1, max_age_days: int = 30) -> int:
        pass

    # ==================== 异步方法 ====================

    async def add_memory_async(self, item: MemoryItem) -> str:
        """异步添加单条记忆 (默认实现，子类可覆写)

        Args:
            item: 记忆对象

        Returns:
            记忆ID，失败返回空字符串
        """
        import asyncio
        return await asyncio.to_thread(self.add_memory, item)

    async def add_memories_batch_async(self, items: List[MemoryItem]) -> List[str]:
        """异步批量添加记忆 (默认实现，子类可覆写)

        Args:
            items: 记忆对象列表

        Returns:
            成功添加的记忆ID列表
        """
        import asyncio
        return await asyncio.to_thread(self.add_memories_batch, items)

    async def search_memory_async(self, query: str, limit: int = 5, user_id: Optional[str] = None, **kwargs) -> List[MemoryItem]:
        """异步搜索记忆 (默认实现，子类可覆写)

        Args:
            query: 搜索查询
            limit: 返回数量限制
            user_id: 可选的用户ID过滤

        Returns:
            匹配的记忆列表
        """
        import asyncio
        return await asyncio.to_thread(self.search_memory, query, limit, user_id, **kwargs)

    # ==================== 工具方法 ====================

    def _calculate_importance(self, content: str,keywords:Optional[List]=None, base_importance: float = 0.5) -> float:
        """计算记忆重要性

        Args:
            content: 记忆内容
            base_importance: 基础重要性

        Returns:
            计算后的重要性分数
        """
        importance = base_importance

        # 基于内容长度
        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = keywords if keywords else ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def get_all_memories(self) -> List[MemoryItem]:
        pass