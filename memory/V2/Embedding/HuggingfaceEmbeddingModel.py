# 使用sklearn的tf-idf
from .BaseEmbeddingModel import BaseEmbeddingModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional

class HuggingfaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.model_name = model_name
        # If model_path is provided, use it. Otherwise use model_name.
        # SentenceTransformer handles both HuggingFace Hub IDs and local paths.
        load_path = model_name
        self.model = SentenceTransformer(load_path)
        self.embedding_size = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: List[str]) -> List[List[float]]:
        return self.model.encode(text).tolist()
    @property
    def dimension(self)->int:
        return self.embedding_size # type: ignore