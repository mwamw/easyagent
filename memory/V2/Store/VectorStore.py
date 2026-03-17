from abc import ABC,abstractmethod
from typing import Any
from ..BaseMemory import MemoryType
class VectorStore(ABC):
    @abstractmethod
    def add_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str]
    ) -> str:
        pass
    
    @abstractmethod
    def remove_vectors(self,ids:list[str]):
        pass

    @abstractmethod
    def clear_type_memory(self,memory_type:MemoryType):
        pass

    @abstractmethod
    def get_collection_stats(self)->dict[str,Any]:
        pass

    @abstractmethod
    def search_similar(self,query_embedding:list[float],where:dict[str,Any],limit:int)->list[dict[str,Any]]:
        pass

    @abstractmethod
    def get_all_vectors(self,with_vector:bool=False)->list[dict[str,Any]]:
        """获取所有向量"""
        pass