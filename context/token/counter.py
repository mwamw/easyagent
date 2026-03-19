"""
Token 计数器

优先使用 tiktoken（精确），fallback 到字符数估算。
"""
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token 计数器，支持多种后端"""

    def __init__(self, model: str = "gpt-4", chars_per_token: float = 3.5):
        """
        Args:
            model: tiktoken 编码对应的模型名
            chars_per_token: 字符估算时的平均字符/token 比率
                             中文约 1.5~2 字符/token，英文约 4 字符/token，
                             混合取 3.5 作默认
        """
        self.chars_per_token = chars_per_token
        self._encoder = None
        self._use_tiktoken = False

        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(model)
            self._use_tiktoken = True
            logger.debug("TokenCounter 使用 tiktoken (%s)", model)
        except Exception:
            logger.debug("tiktoken 不可用，回退到字符估算 (%.1f chars/token)", chars_per_token)

    def count(self, text: str) -> int:
        """计算文本的 token 数"""
        if not text:
            return 0
        if self._use_tiktoken:
            return len(self._encoder.encode(text))
        return max(1, int(len(text) / self.chars_per_token))

    def count_messages(self, messages: List[dict]) -> int:
        """计算消息列表的总 token 数（含角色标记开销）"""
        total = 0
        for msg in messages:
            total += 4  # 每条消息 ~4 token 开销（role + formatting）
            total += self.count(msg.get("content", ""))
        total += 2  # reply priming
        return total

    def truncate(self, text: str, max_tokens: int) -> str:
        """截断文本到指定 token 数"""
        if self.count(text) <= max_tokens:
            return text
        if self._use_tiktoken:
            tokens = self._encoder.encode(text)[:max_tokens]
            return self._encoder.decode(tokens)
        # 字符估算截断
        max_chars = int(max_tokens * self.chars_per_token)
        return text[:max_chars]
