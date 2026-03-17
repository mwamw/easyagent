"""
RAG 管线 - 编排完整的检索增强生成流程
"""
import os
import logging
from typing import List, Optional, Dict, Any

from .document import Document

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG 管线

    编排文档加载、分块、嵌入、存储、检索和生成的完整流程。
    所有组件均可替换，支持从简单到高级的各种配置。

    Args:
        llm: LLM 实例（需实现 invoke 方法）
        embedding: 嵌入模型实例
        vectorstore: 向量存储实例
        loader: 文档加载器（默认 DocumentLoader）
        chunker: 分块器（默认 RecursiveCharacterChunker）
        retriever: 检索器（默认 VectorRetriever）
        query_transformer: 查询转换器（可选）
        prompt_template: 生成提示词模板（需包含 {context} 和 {question}）
        k: 默认检索文档块数

    Example (简单模式):
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

    Example (高级模式):
        >>> from rag.chunker import SemanticChunker
        >>> from rag.retriever import HybridRetriever, VectorRetriever, BM25Retriever
        >>> from rag.query_transform import HyDETransformer
        >>>
        >>> vec_ret = VectorRetriever(vectorstore=store, embedding=emb)
        >>> bm25_ret = BM25Retriever(chunks=chunks)
        >>> hybrid = HybridRetriever(vec_ret, bm25_ret)
        >>>
        >>> pipeline = RAGPipeline(
        ...     llm=llm,
        ...     embedding=emb,
        ...     vectorstore=store,
        ...     chunker=SemanticChunker(emb),
        ...     retriever=hybrid,
        ...     query_transformer=HyDETransformer(llm),
        ... )
    """

    DEFAULT_PROMPT = (
        "你是一个知识问答助手，请根据提供的参考资料回答用户问题。\n\n"
        "## 参考资料\n{context}\n\n"
        "## 回答要求\n"
        "1. 基于参考资料中的信息回答问题\n"
        "2. 如果参考资料中没有相关信息，请明确说明\n"
        "3. 引用信息时，尽量指出来源\n"
        "4. 保持回答的准确性和客观性\n\n"
        "用户问题：{question}"
    )

    def __init__(
        self,
        llm,
        embedding,
        vectorstore,
        loader=None,
        chunker=None,
        retriever=None,
        query_transformer=None,
        prompt_template: Optional[str] = None,
        k: int = 4,
    ):
        self.llm = llm
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.k = k
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.query_transformer = query_transformer

        # 默认加载器
        if loader:
            self.loader = loader
        else:
            from .loader import DocumentLoader
            self.loader = DocumentLoader()

        # 默认分块器
        if chunker:
            self.chunker = chunker
        else:
            from .chunker import RecursiveCharacterChunker
            self.chunker = RecursiveCharacterChunker()

        # 默认检索器
        if retriever:
            self.retriever = retriever
        else:
            from .retriever import VectorRetriever
            self.retriever = VectorRetriever(
                vectorstore=vectorstore, embedding=embedding, k=k,
            )

    def ingest(self, documents: List[Document]) -> List:
        """
        将文档列表导入管线（分块 → 嵌入 → 存储）

        Args:
            documents: 文档列表

        Returns:
            生成的文档块列表
        """
        all_chunks = self.chunker.split_batch(documents)

        if not all_chunks:
            logger.warning("没有生成任何文档块")
            return []

        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding.embed_documents(texts)

        self.vectorstore.add_documents(all_chunks, embeddings)
        logger.info(f"已导入 {len(all_chunks)} 个文档块（来自 {len(documents)} 个文档）")

        return all_chunks

    def ingest_from_path(self, path: str) -> List:
        """
        从文件或目录加载并导入

        Args:
            path: 文件或目录路径

        Returns:
            生成的文档块列表
        """
        if os.path.isdir(path):
            documents = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        doc = self.loader.load(file_path)
                        if doc and doc.content:
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"加载文件失败 {file_path}: {e}")
            return self.ingest(documents) if documents else []
        else:
            doc = self.loader.load(path)
            return self.ingest([doc]) if doc and doc.content else []

    def query(self, question: str, k: Optional[int] = None) -> str:
        """
        查询管线，返回生成的回答

        Args:
            question: 用户问题
            k: 检索文档块数（可选）

        Returns:
            生成的回答文本
        """
        k = k or self.k

        # 查询转换
        search_query = question
        if self.query_transformer:
            search_query = self.query_transformer.transform(question)

        # 检索
        chunks = self.retriever.retrieve(search_query, k=k)

        if not chunks:
            return "未找到相关参考资料，无法回答该问题。"

        # 构建上下文
        context = "\n\n".join([
            f"[来源: {chunk.document_path or '未知'}]\n{chunk.content}"
            for chunk in chunks
        ])

        # 生成
        prompt = self.prompt_template.format(context=context, question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        return response

    def query_with_sources(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        查询管线，返回回答及来源

        Args:
            question: 用户问题
            k: 检索文档块数（可选）

        Returns:
            包含 answer、sources、chunks 的字典
        """
        k = k or self.k

        search_query = question
        if self.query_transformer:
            search_query = self.query_transformer.transform(question)

        chunks = self.retriever.retrieve(search_query, k=k)

        if not chunks:
            return {
                "answer": "未找到相关参考资料，无法回答该问题。",
                "sources": [],
                "chunks": [],
            }

        context = "\n\n".join([
            f"[来源: {chunk.document_path or '未知'}]\n{chunk.content}"
            for chunk in chunks
        ])

        prompt = self.prompt_template.format(context=context, question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        sources = list(set(
            chunk.document_path for chunk in chunks if chunk.document_path
        ))

        return {
            "answer": response,
            "sources": sources,
            "chunks": chunks,
        }
    def get_retriever_results(self, question: str, k: Optional[int] = None) -> str:
        """
        仅获取检索结果（不生成回答）

        Args:
            question: 用户问题
            k: 检索文档块数（可选）

        Returns:
            检索到的文档块内容拼接字符串
        """
        k = k or self.k

        search_query = question
        if self.query_transformer:
            search_query = self.query_transformer.transform(question)

        chunks = self.retriever.retrieve(search_query, k=k)

        if not chunks:
            return "未找到相关参考资料。"

        context = "\n\n".join([
            f"[来源: {chunk.document_path or '未知'}]\n{chunk.content}"
            for chunk in chunks
        ])

        return context