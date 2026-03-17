# Embedding module for EasyAgent V2 memory
from .BaseEmbeddingModel import BaseEmbeddingModel
from .HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel

__all__ = ["BaseEmbeddingModel", "HuggingfaceEmbeddingModel"]
