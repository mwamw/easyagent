# Embedding module for EasyAgent V2 memory
from .BaseEmbeddingModel import BaseEmbeddingModel

__all__ = ["BaseEmbeddingModel", "HuggingfaceEmbeddingModel"]


def __getattr__(name):
    if name == "HuggingfaceEmbeddingModel":
        from .HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
        return HuggingfaceEmbeddingModel
    raise AttributeError(name)
