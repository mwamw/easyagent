from abc import ABC,abstractmethod
from typing import List,Any
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed(self,text:List[str])->List[List[float]]:
        pass