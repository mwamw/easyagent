"""查询转换器基类"""
from abc import ABC, abstractmethod


class BaseQueryTransformer(ABC):
    """
    查询转换器抽象基类

    用于在检索前对用户查询进行转换/增强。

    Example:
        >>> class MyTransformer(BaseQueryTransformer):
        ...     def transform(self, query):
        ...         return query + " 详细解释"
    """

    @abstractmethod
    def transform(self, query: str) -> str:
        """
        转换查询

        Args:
            query: 原始查询

        Returns:
            转换后的查询
        """
        pass
