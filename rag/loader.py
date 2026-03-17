"""
文档加载器模块

支持 30+ 文件格式，基于 MarkItDown 进行统一转换。
"""
from typing import List, Optional
import os
import uuid
import logging

from .document import Document

logger = logging.getLogger(__name__)


def markitdown_support_format():
    supported_formats = {
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Text formats
        '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
        # Images (OCR + metadata)
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp',
        # Audio (transcription + metadata)
        '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg',
        # Archives
        '.zip', '.tar', '.gz', '.rar',
        # Code files
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.scss',
        # Other text
        '.log', '.conf', '.ini', '.cfg', '.yaml', '.yml', '.toml'
    }
    return supported_formats


class DocumentLoader:
    """
    文档加载器

    基于 MarkItDown 支持 30+ 文件格式的统一加载，将各种格式转换为 Markdown 文本。
    可选集成 LLM 用于图片 OCR、音频转录等高级功能。

    Args:
        support_format: 支持的文件格式集合
        api_key: LLM API 密钥（用于高级 OCR/转录）
        api_url: LLM API 地址
        llm_model: LLM 模型名称

    Example:
        >>> loader = DocumentLoader()
        >>> doc = loader.load("report.pdf")
        >>> print(doc.content)
    """

    def __init__(
        self,
        support_format: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        llm_model: Optional[str] = "gemini-2.5-flash",
    ):
        self.support_format = support_format or markitdown_support_format()
        self.api_key = api_key
        self.api_url = api_url
        self.llm_model = llm_model

        from markitdown import MarkItDown

        if self.api_key and self.api_url:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            self.md_instance = MarkItDown(llm_client=client, llm_model=self.llm_model)
        else:
            self.md_instance = MarkItDown()

    def detect_document_type(self, file_path: str) -> str:
        """检测文档类型"""
        file_suffix = os.path.splitext(file_path)[-1].lower()
        if file_suffix in self.support_format:
            return file_suffix
        else:
            raise ValueError(f"不支持的文档类型: {file_suffix}")

    def _is_image(self, file_path: str) -> bool:
        file_suffix = os.path.splitext(file_path)[-1].lower()
        return file_suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']

    def _is_audio(self, file_path: str) -> bool:
        file_suffix = os.path.splitext(file_path)[-1].lower()
        return file_suffix in ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']

    def _is_video(self, file_path: str) -> bool:
        file_suffix = os.path.splitext(file_path)[-1].lower()
        return file_suffix in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    def _is_pdf(self, file_path: str) -> bool:
        file_suffix = os.path.splitext(file_path)[-1].lower()
        return file_suffix == '.pdf'

    def _is_table(self, file_path: str) -> bool:
        file_suffix = os.path.splitext(file_path)[-1].lower()
        return file_suffix in ['.xls', '.xlsx', '.csv']

    def load(self, file_path: str) -> Document:
        """
        加载文件并转换为 Document

        Args:
            file_path: 文件路径

        Returns:
            Document 实例
        """
        if self._is_pdf(file_path):
            return self._load_pdf(file_path)
        elif self._is_image(file_path):
            return self._load_image(file_path)
        elif self._is_audio(file_path):
            return self._load_audio(file_path)
        elif self._is_video(file_path):
            return self._load_video(file_path)
        elif self._is_table(file_path):
            return self._load_table(file_path)
        return self._load_text(file_path)

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """
        批量加载目录下的所有文件

        Args:
            dir_path: 目录路径
            recursive: 是否递归加载子目录

        Returns:
            Document 列表
        """
        documents = []
        if recursive:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        doc = self.load(file_path)
                        if doc and doc.content:
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"加载文件失败 {file_path}: {e}")
        else:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    try:
                        doc = self.load(file_path)
                        if doc and doc.content:
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"加载文件失败 {file_path}: {e}")

        logger.info(f"从 {dir_path} 加载了 {len(documents)} 个文档")
        return documents

    def _convert_with_markitdown(self, file_path: str, doc_type: str) -> Document:
        """使用 MarkItDown 统一转换"""
        result = self.md_instance.convert(file_path)
        return Document(
            document_id=str(uuid.uuid4()),
            document_path=file_path,
            content=result.text_content or "",
            metadata={"source": file_path, "type": doc_type},
            document_type=doc_type,
        )

    def _load_pdf(self, file_path: str) -> Document:
        """加载 PDF 文件"""
        return self._convert_with_markitdown(file_path, "pdf")

    def _load_image(self, file_path: str) -> Document:
        """加载图片文件（OCR / 描述）"""
        return self._convert_with_markitdown(file_path, "image")

    def _load_audio(self, file_path: str) -> Document:
        """加载音频文件（转录）"""
        return self._convert_with_markitdown(file_path, "audio")

    def _load_video(self, file_path: str) -> Document:
        """加载视频文件"""
        return self._convert_with_markitdown(file_path, "video")

    def _load_table(self, file_path: str) -> Document:
        """加载表格文件"""
        return self._convert_with_markitdown(file_path, "table")

    def _load_text(self, file_path: str) -> Document:
        """加载文本文件"""
        try:
            return self._convert_with_markitdown(file_path, "text")
        except Exception:
            # 降级方案：直接读取文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return Document(
                document_id=str(uuid.uuid4()),
                document_path=file_path,
                content=content,
                metadata={"source": file_path, "type": "text"},
                document_type="text",
            )

