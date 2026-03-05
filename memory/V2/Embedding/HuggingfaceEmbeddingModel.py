# 使用sklearn的tf-idf
from .BaseEmbeddingModel import BaseEmbeddingModel
from sentence_transformers import SentenceTransformer
from typing import List

class HuggingfaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_size = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: List[str]) -> List[List[float]]:
        return self.model.encode(text).tolist()
    @property
    def dimension(self)->int:
        return self.embedding_size # type: ignore