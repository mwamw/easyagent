"""
EasyAgent RAG 模块

模块化的检索增强生成（Retrieval-Augmented Generation）框架。
所有组件均可独立使用或自由组合，支持从简单到高级的各种 RAG 策略。

模块结构:
    - document: 文档和文档块数据模型
    - loader: 文档加载器（支持 30+ 格式）
    - chunker: 文本分块策略（固定/递归/语义/Token）
    - embedding: 嵌入模型（OpenAI/HuggingFace）
    - vectorstore: 向量存储（内存/ChromaDB）
    - retriever: 检索策略（向量/BM25/混合/多查询/重排序/压缩）
    - query_transform: 查询转换（HyDE/Step-Back）
    - pipeline: RAG 管线编排器

快速开始（简单模式）:
    >>> from rag import RAGPipeline
    >>> from rag.embedding import OpenAIEmbedding
    >>> from rag.vectorstore import MemoryVectorStore
    >>> from core.llm import EasyLLM
    >>>
    >>> pipeline = RAGPipeline(
    ...     llm=EasyLLM(model="gpt-4"),
    ...     embedding=OpenAIEmbedding(),
    ...     vectorstore=MemoryVectorStore(),
    ... )
    >>> pipeline.ingest_from_path("./docs/")
    >>> answer = pipeline.query("什么是RAG?")

高级模式（自由组合组件）:
    >>> from rag.chunker import SemanticChunker
    >>> from rag.retriever import HybridRetriever, VectorRetriever, BM25Retriever
    >>> from rag.query_transform import HyDETransformer
    >>>
    >>> pipeline = RAGPipeline(
    ...     llm=llm, embedding=emb, vectorstore=store,
    ...     chunker=SemanticChunker(emb),
    ...     retriever=HybridRetriever(VectorRetriever(store, emb), BM25Retriever()),
    ...     query_transformer=HyDETransformer(llm),
    ... )
"""

# 核心数据模型
from .document import Document, Document_Chunk

# 文档加载
from .loader import DocumentLoader

# 文本分块
from .chunker import (
    BaseChunker,
    FixedChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    TokenChunker,
)

# 嵌入模型
from .embedding import BaseEmbedding, OpenAIEmbedding, HuggingFaceEmbedding

# 向量存储
from .vectorstore import BaseVectorStore, MemoryVectorStore, ChromaVectorStore

# 检索器
from .retriever import (
    BaseRetriever,
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    MultiQueryRetriever,
    ReRankRetriever,
    CompressionRetriever,
)

# 查询转换
from .query_transform import BaseQueryTransformer, HyDETransformer, StepBackTransformer

# 管线
from .pipeline import RAGPipeline

__all__ = [
    # 数据模型
    "Document",
    "Document_Chunk",
    # 加载器
    "DocumentLoader",
    # 分块器
    "BaseChunker",
    "FixedChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "TokenChunker",
    # 嵌入
    "BaseEmbedding",
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    # 向量存储
    "BaseVectorStore",
    "MemoryVectorStore",
    "ChromaVectorStore",
    # 检索器
    "BaseRetriever",
    "VectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "ReRankRetriever",
    "CompressionRetriever",
    # 查询转换
    "BaseQueryTransformer",
    "HyDETransformer",
    "StepBackTransformer",
    # 管线
    "RAGPipeline",
]
