"""
EasyLLM - 统一的 LLM 接口

提供对不同 LLM 服务的统一访问接口。
"""
from .Message import Message
from .providers import create_provider, BaseProvider
from typing import Optional, Any, Generator
import logging
import os

logger = logging.getLogger(__name__)


class EasyLLM:
    """
    统一的 LLM 接口类
    
    支持多种 LLM 服务：
    - OpenAI (GPT-4, GPT-3.5)
    - Google (Gemini)
    - Anthropic (Claude)
    - DeepSeek
    - Qwen (通义千问)
    - Kimi (Moonshot)
    - 智谱 AI
    - Ollama
    - vLLM
    - 其他 OpenAI API 兼容服务
    
    示例:
        >>> llm = EasyLLM(model="gpt-4")
        >>> response = llm.invoke([{"role": "user", "content": "Hello"}])
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        provide: Optional[str] = "auto",
        **kwargs
    ):
        """
        初始化 EasyLLM
        
        Args:
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            api_key: API 密钥
            base_url: API 地址
            timeout: 超时时间
            provide: Provider 类型 (auto, openai, google, anthropic, ...)
        """
        self.provide = provide
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs
        
        # 自动检测 Provider
        if self.provide == "auto":
            self.provide = self._auto_detect_provider()
        
        # 解析 API 密钥和地址
        self.resovle_api_key, self.resovle_base_url = self._resolve_api_key_and_base_url()
        
        # 设置默认模型
        if not self.model:
            self.model = self._get_default_model()
        
        # 验证配置
        if not self.resovle_api_key or not self.resovle_base_url:
            raise ValueError("API密钥和服务地址必须被提供或在.env文件中定义。")
        
        # 创建 Provider
        self._provider: BaseProvider = create_provider(
            provider_name=self.provide,
            model=self.model,
            api_key=self.resovle_api_key,
            base_url=self.resovle_base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            **kwargs
        )
        
        # 保持向后兼容
        self.client = self._provider.client
        
        logger.info(f"EasyLLM 初始化完成: provider={self.provide}, model={self.model}")
    
    @property
    def provider(self) -> BaseProvider:
        """获取当前 Provider"""
        return self._provider
    
    def _auto_detect_provider(self) -> str:
        """自动检测 Provider 类型"""
        # 1. 根据环境变量判断
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("Google_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "google"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"
        
        # 2. 根据 base_url 判断
        base_url = self.base_url or os.getenv("LLM_BASE_URL") or ""
        base_url_lower = base_url.lower()
        
        if "api.openai.com" in base_url_lower:
            return "openai"
        elif "google" in base_url_lower:
            return "google"
        elif "anthropic" in base_url_lower:
            return "anthropic"
        elif "api.deepseek.com" in base_url_lower:
            return "deepseek"
        elif "dashscope.aliyuncs.com" in base_url_lower:
            return "qwen"
        elif "api-inference.modelscope.cn" in base_url_lower:
            return "modelscope"
        elif "api.moonshot.cn" in base_url_lower:
            return "kimi"
        elif "open.bigmodel.cn" in base_url_lower:
            return "zhipu"
        elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
            if ":11434" in base_url_lower or "ollama" in base_url_lower:
                return "ollama"
            elif ":8000" in base_url_lower:
                return "vllm"
        
        # 3. 根据模型名判断
        if self.model:
            model_lower = self.model.lower()
            if "gpt" in model_lower:
                return "openai"
            elif "gemini" in model_lower:
                return "google"
            elif "claude" in model_lower:
                return "anthropic"
            elif "deepseek" in model_lower:
                return "deepseek"
            elif "qwen" in model_lower:
                return "qwen"
            elif "moonshot" in model_lower or "kimi" in model_lower:
                return "kimi"
            elif "glm" in model_lower or "chatglm" in model_lower:
                return "zhipu"
        
        # 4. 根据 API Key 格式判断
        api_key = self.api_key or os.getenv("LLM_API_KEY") or ""
        if api_key:
            api_key_lower = api_key.lower()
            if "." in api_key_lower[-20:]:
                return "zhipu"
        
        return "openai"  # 默认使用 OpenAI 兼容
    
    def _get_default_model(self) -> str:
        """获取默认模型名称"""
        default_models = {
            "openai": "gpt-3.5-turbo",
            "google": "gemini-2.5-flash",
            "anthropic": "claude-3-5-sonnet",
            "deepseek": "deepseek-chat",
            "qwen": "qwen-plus",
            "modelscope": "Qwen/Qwen2.5-VL-72B-Instruct",
            "kimi": "moonshot-v1-8k",
            "zhipu": "glm-4",
            "ollama": "llama3",
            "vllm": "llama3",
        }
        return default_models.get(self.provide, "gpt-3.5-turbo")
    
    def _resolve_api_key_and_base_url(self) -> tuple[str, str]:
        """解析 API 密钥和地址"""
        if self.api_key and self.base_url:
            return self.api_key, self.base_url
        
        provider_configs = {
            "openai": (
                os.getenv("OPENAI_API_KEY"),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            ),
            "google": (
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
            ),
            "anthropic": (
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            ),
            "deepseek": (
                os.getenv("DEEPSEEK_API_KEY"),
                os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            ),
            "qwen": (
                os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
                os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            ),
            "modelscope": (
                os.getenv("MODELSCOPE_API_KEY"),
                os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1/")
            ),
            "kimi": (
                os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"),
                os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
            ),
            "zhipu": (
                os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"),
                os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
            ),
            "ollama": (
                os.getenv("OLLAMA_API_KEY", "ollama"),
                os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            ),
            "vllm": (
                os.getenv("VLLM_API_KEY", "vllm"),
                os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            ),
        }
        
        if self.provide in provider_configs:
            env_key, env_url = provider_configs[self.provide]
            return (
                self.api_key or env_key or os.getenv("LLM_API_KEY", ""),
                self.base_url or env_url or os.getenv("LLM_BASE_URL", "")
            )
        
        return (
            self.api_key or os.getenv("LLM_API_KEY", ""),
            self.base_url or os.getenv("LLM_BASE_URL", "")
        )
    
    # ==================== 主要 API ====================
    
    def invoke(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        同步调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            LLM 响应内容
        """
        messages = self._convert_messages(messages)
        return self._provider.invoke(messages, temperature=temperature, **kwargs)
    
    def stream(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Yields:
            响应内容片段
        """
        messages = self._convert_messages(messages)
        yield from self._provider.stream(messages, temperature=temperature, **kwargs)
    
    def invoke_with_tools(
        self,
        messages: list[dict[str, str] | Message],
        tools: list[dict],
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        带工具调用的 LLM 调用
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            temperature: 温度参数
            
        Returns:
            LLM 响应对象
        """
        messages = self._convert_messages(messages)
        return self._provider.invoke_with_tools(messages, tools, temperature=temperature, **kwargs)
    
    def format_tool_result(
        self,
        content: str,
        tool_id: str,
        tool_name: str
    ) -> dict:
        """
        格式化工具结果为当前 Provider 需要的格式
        
        Args:
            content: 工具执行结果
            tool_id: 工具调用 ID
            tool_name: 工具名称
            
        Returns:
            格式化后的消息字典
        """
        return self._provider.format_tool_result(content, tool_id, tool_name)
    
    # ==================== 向后兼容的方法 ====================
    
    def think(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """流式输出（向后兼容）"""
        messages = self._convert_messages(messages)
        for chunk in self._provider.stream(messages, temperature=temperature):
            print(chunk, end="", flush=True)
            yield chunk
    
    def stream_invoke(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """流式调用（向后兼容）"""
        yield from self.think(messages, temperature)
    
    def get_client(self):
        """获取底层客户端（向后兼容）"""
        return self.client
    
    # ==================== 辅助方法 ====================
    
    def _convert_messages(self, messages: list) -> list[dict]:
        """将 Message 对象转换为字典"""
        return [
            msg.to_dict() if isinstance(msg, Message) else msg
            for msg in messages
        ]
    
    def get_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容"""
        return self._provider.get_thinking_content(response)
    
    def has_tool_calls(self, response: Any) -> bool:
        """检查是否有工具调用"""
        return self._provider.has_tool_calls(response)
    
    def get_tool_calls(self, response: Any) -> list:
        """获取工具调用列表"""
        return self._provider.get_tool_calls(response)
    
    def create_client(self):
        """创建客户端（向后兼容）"""
        return self._provider.client
