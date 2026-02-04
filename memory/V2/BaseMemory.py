from abc import ABC, abstractmethod
from datetime import datetime
import enum
from optparse import Option
from sqlite3.dbapi2 import Timestamp
from numpy._typing import _128Bit
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

class MemoryConfig(BaseModel):

    #基础设置
    max_capacity:int=100
    importance_threshold:float=0.1
    decay_factor:float=0.95

    #工作记忆配置
    max_working_token:int=10000



class BaseMemory(ABC):
    """记忆基类
    需要提供一下功能
    - 添加记忆
    - 删除记忆
    - 更新记忆
    - 随机生成记忆id
    - 根据查询搜索记忆
    - 查找是否存在记忆
    - 清除记忆
    - 获取记忆统计信息
    - 计算记忆重要性
    """

    def __init__(self,config:MemoryConfig):
        self.config=config
        self.memory_type=self.__class__.__name__.replace("memory","")

    @abstractmethod
    def add_memory(self,item:MemoryItem)->str:
        """
        添加记忆

        Args:
            memory_item:记忆对象
        """

        
        pass


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
    def search_memory(self,query:str,limit:int=5,user_id:Optional[str]=None,**kwargs):
        pass

    @abstractmethod
    def find_memory(self,id:str)->bool:
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