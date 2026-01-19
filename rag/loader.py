"""
文档加载器模块
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Iterator
from pathlib import Path
import os
import logging

from .document import Document

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """
    文档加载器抽象基类
    
    所有文档加载器必须实现 load 方法。
    """
    
    @abstractmethod
    def load(self) -> List[Document]:
        """加载文档"""
        pass
    
    def lazy_load(self) -> Iterator[Document]:
        """延迟加载文档（默认实现）"""
        yield from self.load()


class TextLoader(DocumentLoader):
    """
    文本文件加载器
    
    加载 .txt 等纯文本文件。
    
    Example:
        >>> loader = TextLoader("document.txt")
        >>> docs = loader.load()
    """
    
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        autodetect_encoding: bool = False
    ):
        """
        初始化文本加载器
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            autodetect_encoding: 是否自动检测编码
        """
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
    
    def load(self) -> List[Document]:
        """加载文本文件"""
        path = Path(self.file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        encoding = self.encoding
        if self.autodetect_encoding:
            encoding = self._detect_encoding()
        
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试使用其他编码
            for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
                try:
                    with open(path, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"无法解码文件: {self.file_path}")
        
        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "file_size": path.stat().st_size,
        }
        
        return [Document(page_content=content, metadata=metadata)]
    
    def _detect_encoding(self) -> str:
        """检测文件编码"""
        try:
            import chardet
            with open(self.file_path, "rb") as f:
                result = chardet.detect(f.read())
                return result.get("encoding", "utf-8")
        except ImportError:
            logger.warning("chardet 未安装，使用默认编码 utf-8")
            return "utf-8"


class PDFLoader(DocumentLoader):
    """
    PDF 文件加载器
    
    加载 PDF 文件并提取文本内容。
    需要安装 pypdf 库。
    
    Example:
        >>> loader = PDFLoader("document.pdf")
        >>> docs = loader.load()
    """
    
    def __init__(
        self,
        file_path: str,
        extract_images: bool = False
    ):
        """
        初始化 PDF 加载器
        
        Args:
            file_path: PDF 文件路径
            extract_images: 是否提取图片（暂不支持）
        """
        self.file_path = file_path
        self.extract_images = extract_images
    
    def load(self) -> List[Document]:
        """加载 PDF 文件"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("PDFLoader 需要 pypdf 库。请运行: pip install pypdf")
        
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        reader = PdfReader(self.file_path)
        documents = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                metadata = {
                    "source": str(path.absolute()),
                    "filename": path.name,
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        logger.info(f"从 PDF 加载了 {len(documents)} 页")
        return documents


class DirectoryLoader(DocumentLoader):
    """
    目录加载器
    
    加载目录下的所有文件。
    
    Example:
        >>> loader = DirectoryLoader("./docs", glob="**/*.txt")
        >>> docs = loader.load()
    """
    
    def __init__(
        self,
        path: str,
        glob: str = "**/*.*",
        loader_cls: type = TextLoader,
        loader_kwargs: Optional[dict] = None,
        recursive: bool = True,
        show_progress: bool = False
    ):
        """
        初始化目录加载器
        
        Args:
            path: 目录路径
            glob: 文件匹配模式
            loader_cls: 文件加载器类
            loader_kwargs: 传递给加载器的参数
            recursive: 是否递归加载子目录
            show_progress: 是否显示进度
        """
        self.path = path
        self.glob = glob
        self.loader_cls = loader_cls
        self.loader_kwargs = loader_kwargs or {}
        self.recursive = recursive
        self.show_progress = show_progress
    
    def load(self) -> List[Document]:
        """加载目录下的所有文件"""
        path = Path(self.path)
        if not path.is_dir():
            raise ValueError(f"不是有效目录: {self.path}")
        
        documents = []
        files = list(path.glob(self.glob))
        
        for file_path in files:
            if file_path.is_file():
                try:
                    loader = self.loader_cls(str(file_path), **self.loader_kwargs)
                    docs = loader.load()
                    documents.extend(docs)
                    if self.show_progress:
                        logger.info(f"已加载: {file_path.name}")
                except Exception as e:
                    logger.warning(f"加载文件失败 {file_path}: {e}")
        
        logger.info(f"从目录加载了 {len(documents)} 个文档")
        return documents
