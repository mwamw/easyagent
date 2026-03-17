from abc import ABC,abstractmethod
from typing import Dict,Any,Optional
from ..BaseMemory import MemoryItem, MemoryType
class DocumentStore(ABC):
    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        pass    


    @abstractmethod
    def remove_memory(self,memory_id:str):
        pass

    @abstractmethod
    def update_memory(self,memory_id:str,content:str,importance:Optional[float]=None,properties:Optional[dict[str,Any]]=None):
        pass
    
    @abstractmethod
    def get_memory(self,memory_id:str)->Optional[MemoryItem]:
        pass

    @abstractmethod
    def clear_type_memory(self,memory_type:MemoryType):
        pass
    
    @abstractmethod
    def get_database_stats(self)->dict[str,Any]:
        pass

    @abstractmethod
    def search_memory(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 1000
    ) -> list[MemoryItem]:
        pass

    @abstractmethod
    def get_all_memories(self)->list[MemoryItem]:
        pass

    @abstractmethod
    def clear_all(self):
        pass