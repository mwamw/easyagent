"""Step-Back 查询转换器（高级）"""
import logging
from typing import Any

from .base import BaseQueryTransformer

logger = logging.getLogger(__name__)


class StepBackTransformer(BaseQueryTransformer):
    """
    Step-Back 查询转换器

    将具体的用户问题转换为更抽象、更通用的问题，以便检索到更广泛的
    背景知识。适用于需要深层背景知识才能回答的问题。

    参考论文: Zheng et al., "Take a Step Back: Evoking Reasoning via Abstraction"

    Args:
        llm: LLM 实例（需实现 invoke 方法）

    Example:
        >>> transformer = StepBackTransformer(llm)
        >>> # "Python 3.12 有什么新特性?" -> "Python 的版本演进历史是什么?"
        >>> abstract_query = transformer.transform("Python 3.12 有什么新特性?")
    """

    def __init__(self, llm: Any):
        self.llm = llm

    def transform(self, query: str) -> str:
        prompt = (
            "请将以下具体问题转换为一个更通用、更抽象的问题，"
            "以便更好地检索背景知识。只返回转换后的问题。\n\n"
            f"具体问题：{query}\n\n"
            "抽象问题："
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.strip()
        except Exception as e:
            logger.warning(f"Step-Back 转换失败: {e}")
            return query
