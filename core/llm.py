"""
EasyLLM - 统一的 LLM 接口

提供对不同 LLM 服务的统一访问接口。
"""
from .Message import Message
from .history import CanonicalMessage
from .providers import BaseProvider, BaseProviderCodec, create_codec, create_provider, provider_requires_base_url
from .request_input import ReplayRequestInput
from typing import Optional, Any, Generator, AsyncGenerator
import logging
import os

logger = logging.getLogger(__name__)


def _read_usage_value(payload: Any, *keys: str) -> Any:
    for key in keys:
        if payload is None:
            return None
        if isinstance(payload, dict):
            if key in payload and payload[key] is not None:
                return payload[key]
            continue
        value = getattr(payload, key, None)
        if value is not None:
            return value
    return None


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
        provider: Optional[str] = "auto",
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
        provider: Provider 类型 (auto, openai, openai_responses, google, google_native, anthropic, anthropic_native, ...)
        """
        self.provider_name = provider
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs
        
        # 自动检测 Provider
        if self.provider_name == "auto":
            self.provider_name = self._auto_detect_provider()
        
        # 解析 API 密钥和地址
        self.resolve_api_key, self.resolve_base_url = self._resolve_api_key_and_base_url()
        
        # 设置默认模型
        if not self.model:
            self.model = self._get_default_model()
        
        # 验证配置
        if not self.resolve_api_key:
            raise ValueError("API密钥必须被提供或在.env文件中定义。")
        if provider_requires_base_url(self.provider_name or "") and not self.resolve_base_url:
            raise ValueError("服务地址必须被提供或在.env文件中定义。")
        
        # 创建 Provider
        self._provider: BaseProvider = create_provider(
            provider_name=self.provider_name, # type: ignore
            model=self.model,
            api_key=self.resolve_api_key,
            base_url=self.resolve_base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            **kwargs
        )
        self.codec: BaseProviderCodec = create_codec(self.provider_name)
        
        # 保持向后兼容
        self.client = self._provider.client
        
        logger.info(f"EasyLLM 初始化完成: provider={self.provider_name}, model={self.model}")
    
    @property
    def provider(self) -> BaseProvider:
        """获取当前 Provider"""
        return self._provider

    def _get_codec(self) -> BaseProviderCodec:
        codec = getattr(self, "codec", None)
        provider_name = getattr(self, "provider_name", None)
        if codec is None or getattr(codec, "provider_name", None) != provider_name:
            codec = create_codec(provider_name)
            self.codec = codec
        return codec

    def export_tools(self, tools: Any) -> Any:
        if tools is None:
            return None
        if hasattr(tools, "get_visible_tools") and callable(getattr(tools, "get_visible_tools")):
            return self._provider.build_tool_payload(tools.get_visible_tools())
        if hasattr(tools, "get_spec") and callable(getattr(tools, "get_spec")):
            return self._provider.build_tool_payload([tools])
        if isinstance(tools, (list, tuple)):
            items = list(tools)
            if not items:
                return self._provider.build_tool_payload(items)
            if all(hasattr(item, "get_spec") and callable(getattr(item, "get_spec")) for item in items):
                return self._provider.build_tool_payload(items)
            return items
        return tools
    
    def _auto_detect_provider(self) -> str:
        """自动检测 Provider 类型"""
        # 1. 根据环境变量判断
        if os.getenv("OPENAI_API_KEY"):
            return "openai_responses"
        if os.getenv("Google_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "google_native"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic_native"
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
            return "openai_responses"
        elif "google" in base_url_lower:
            return "google_native"
        elif "anthropic" in base_url_lower:
            return "anthropic_native"
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
                return "openai_responses"
            elif "gemini" in model_lower:
                return "google_native"
            elif "claude" in model_lower:
                return "anthropic_native"
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
            "openai_responses": "gpt-4o",
            "google": "gemini-2.5-pro",
            "google_native": "gemini-2.5-pro",
            "anthropic": "claude-4.5-sonnet",
            "anthropic_native": "claude-4.5-sonnet",
            "deepseek": "deepseek-chat",
            "qwen": "qwen3-32b",
            "modelscope": "Qwen/Qwen2.5-VL-72B-Instruct",
            "kimi": "moonshot-v1-8k",
            "zhipu": "glm-4",
            "ollama": "llama3",
            "vllm": "llama3",
        }
        return default_models.get(self.provider_name, "gpt-3.5-turbo") # type: ignore
    
    def _resolve_api_key_and_base_url(self) -> tuple[str, str]:
        """解析 API 密钥和地址"""
        if self.api_key and self.base_url:
            return self.api_key, self.base_url
        
        provider_configs = {
            "openai": (
                os.getenv("OPENAI_API_KEY"),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            ),
            "openai_responses": (
                os.getenv("OPENAI_API_KEY"),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            ),
            "google": (
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("GOOGLE_BASE_URL", "")
            ),
            "google_native": (
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("GOOGLE_BASE_URL", "")
            ),
            "anthropic": (
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            ),
            "anthropic_native": (
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("ANTHROPIC_BASE_URL", "")
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
        
        if self.provider_name in provider_configs:
            env_key, env_url = provider_configs[self.provider_name]
            return (
                self.api_key or env_key or os.getenv("LLM_API_KEY", ""),
                self.base_url or env_url or os.getenv("LLM_BASE_URL", "")
            )
        
        return (
            self.api_key or os.getenv("LLM_API_KEY", ""),
            self.base_url or os.getenv("LLM_BASE_URL", "")
        )
    
    # ==================== 主要 API ====================

    def _build_provider_request(
        self,
        request_input: ReplayRequestInput,
        *,
        tools: Any = None,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        request = self._provider.build_request(
            request_input.replay_history,
            system_prompt=request_input.render_system_prompt(),
            tools=tools,
            temperature=temperature,
            reasoning=reasoning,
            stream=stream,
            **kwargs,
        )
        return self._provider.apply_cache_policy(request, request_input)
    
    def invoke(
        self,
        messages: list[dict[str, str]],
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str | None:
        """
        同步调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            LLM 响应内容
        """
        response = self.invoke_raw(messages, reasoning=reasoning, temperature=temperature, **kwargs)
        return self.get_response_content(response)

    def invoke_raw(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """同步调用并返回 provider 原始响应对象。"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        request = self._build_provider_request(
            request_input,
            temperature=temperature,
            reasoning=reasoning,
            stream=False,
            **kwargs,
        )
        return self._provider.invoke_raw(request)
    
    def stream(
        self,
        messages: list[dict[str, str]]|ReplayRequestInput,
        reasoning: Optional[dict[str, Any]] = None,
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
        for event in self.stream_events(messages, reasoning=reasoning, temperature=temperature, **kwargs):
            if event.get("type") == "text_delta":
                yield event.get("delta", "") or ""

    def stream_events(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """流式调用，返回 thinking / text / final 事件。"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        request = self._build_provider_request(
            request_input,
            temperature=temperature,
            reasoning=reasoning,
            stream=True,
            **kwargs,
        )
        raw_stream = self._provider.stream_raw(request)
        yield from self._get_codec().stream_events(raw_stream, tools=False)
    
    def invoke_with_tools(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        tools: Any,
        reasoning: Optional[dict[str, Any]] = None,
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
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        tool_payload = self.export_tools(tools)
        try:
            request = self._build_provider_request(
                request_input,
                tools=tool_payload,
                temperature=temperature,
                reasoning=reasoning,
                stream=False,
                **kwargs,
            )
            return self._provider.invoke_raw(request)
        except Exception as e:
            logger.error(
                f"LLM工具调用失败 当前消息{request_input.replay_history[-1] if request_input.replay_history else None}"
            )
            raise e

    def stream_with_tools(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        tools: Any,
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """同步流式带工具调用，返回统一事件流。"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        tool_payload = self.export_tools(tools)
        request = self._build_provider_request(
            request_input,
            tools=tool_payload,
            reasoning=reasoning,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        raw_stream = self._provider.stream_raw(request)
        yield from self._get_codec().stream_events(raw_stream, tools=True)

    # ==================== 异步 API ====================

    async def ainvoke(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str | None:
        """
        异步调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Returns:
            LLM 响应内容
        """
        response = await self.ainvoke_raw(messages, temperature=temperature, reasoning=reasoning, **kwargs)
        return self.get_response_content(response)

    async def ainvoke_raw(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """异步调用并返回 provider 原始响应对象。"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        request = self._build_provider_request(
            request_input,
            temperature=temperature,
            reasoning=reasoning,
            stream=False,
            **kwargs,
        )
        return await self._provider.async_invoke_raw(request)

    async def astream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        异步流式调用 LLM
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            
        Yields:
            响应内容片段
        """
        async for event in self.astream_events(messages, reasoning=reasoning, temperature=temperature, **kwargs):
            if event.get("type") == "text_delta":
                yield event.get("delta", "") or ""

    async def astream_events(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式调用，返回 thinking / text / final 事件。"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        request = self._build_provider_request(
            request_input,
            reasoning=reasoning,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        raw_stream = await self._provider.async_stream_raw(request)
        async for event in self._get_codec().astream_events(raw_stream, tools=False):
            yield event

    async def ainvoke_with_tools(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        tools: Any,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        异步带工具调用的 LLM 调用
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            temperature: 温度参数
            
        Returns:
            LLM 响应对象
        """
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        tool_payload = self.export_tools(tools)
        try:
            request = self._build_provider_request(
                request_input,
                tools=tool_payload,
                reasoning=reasoning,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            return await self._provider.async_invoke_raw(request)
        except Exception as e:
            logger.error(
                f"LLM 异步工具调用失败 当前消息{request_input.replay_history[-1] if request_input.replay_history else None}"
            )
            raise e

    async def astream_with_tools(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        tools: Any,
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式带工具调用，返回统一事件流。"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        tool_payload = self.export_tools(tools)
        request = self._build_provider_request(
            request_input,
            tools=tool_payload,
            reasoning=reasoning,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        raw_stream = await self._provider.async_stream_raw(request)
        async for event in self._get_codec().astream_events(raw_stream, tools=True):
            yield event

    # ==================== 向后兼容的方法 ====================
    
    def think(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
        temperature: Optional[float] = None ,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """流式输出（向后兼容）"""
        request_input = self._prepare_request_input(messages) # type: ignore[arg-type]
        for chunk in self.stream(request_input, temperature=temperature,reasoning=reasoning):
            print(chunk, end="", flush=True)
            yield chunk
    
    def stream_invoke(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """流式调用（向后兼容）"""
        yield from self.think(messages, temperature,reasoning)
    
    def get_client(self):
        """获取底层客户端（向后兼容）"""
        return self.client

    def close(self) -> None:
        """关闭底层 Provider 客户端。"""
        provider = getattr(self, "_provider", None)
        if provider is not None and hasattr(provider, "close"):
            provider.close()  # type: ignore[misc]

    async def aclose(self) -> None:
        """异步关闭底层 Provider 客户端。"""
        provider = getattr(self, "_provider", None)
        if provider is not None and hasattr(provider, "aclose"):
            await provider.aclose()  # type: ignore[misc]
    
    # ==================== 辅助方法 ====================

    def is_request_ready_message(self, message: Any) -> bool:
        return self._get_codec().is_request_ready_message(message)

    def history_entry_to_canonical(self, message: Any) -> list[CanonicalMessage]:
        return self._get_codec().history_entry_to_canonical(message)

    def query_to_canonical(self, query: str) -> list[CanonicalMessage]:
        return self._get_codec().query_to_canonical(query)

    def query_to_replay(self, query: str) -> list[Any]:
        return self._get_codec().query_to_replay(query)

    def response_to_canonical(self, response: Any, *, include_reasoning: bool = False) -> list[CanonicalMessage]:
        return self._get_codec().response_to_canonical(response, include_reasoning=include_reasoning)

    def response_to_replay(self, response: Any, *, include_reasoning: bool = False) -> list[Any]:
        return self._get_codec().response_to_replay(response, include_reasoning=include_reasoning)

    def assistant_message_to_canonical(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[CanonicalMessage]:
        return self._get_codec().assistant_message_to_canonical(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
        )

    def assistant_message_to_replay(
        self,
        *,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> list[Any]:
        return self._get_codec().assistant_message_to_replay(
            content=content,
            tool_calls=tool_calls,
            thinking=thinking,
        )

    def tool_result_to_canonical(self, content: str, tool_id: str, tool_name: str) -> list[CanonicalMessage]:
        return self._get_codec().tool_result_to_canonical(content, tool_id, tool_name)

    def tool_result_to_replay(self, content: str, tool_id: str, tool_name: str) -> list[Any]:
        return self._get_codec().tool_result_to_replay(content, tool_id, tool_name)

    def canonical_to_replay_history(
        self,
        messages: list[Any],
        provider_name: Optional[str] = None,
    ) -> list[Any]:
        target_provider = provider_name or getattr(self, "provider_name", None)
        if target_provider == getattr(self, "provider_name", None):
            return self._get_codec().canonical_to_replay(messages)
        return create_codec(target_provider).canonical_to_replay(messages)

    def replay_to_canonical_history(
        self,
        messages: list[Any],
        provider_name: Optional[str] = None,
    ) -> list[CanonicalMessage]:
        target_provider = provider_name or getattr(self, "provider_name", None)
        if target_provider == getattr(self, "provider_name", None):
            return self._get_codec().replay_to_canonical(messages)
        return create_codec(target_provider).replay_to_canonical(messages)

    def append_replay_entry(
        self,
        prepared: list[Any],
        entry: Any,
        provider_name: Optional[str] = None,
    ) -> None:
        target_provider = provider_name or getattr(self, "provider_name", None)
        if target_provider == getattr(self, "provider_name", None):
            self._get_codec().append_replay_entry(prepared, entry)
            return
        create_codec(target_provider).append_replay_entry(prepared, entry)

    def count_request_tokens(
        self,
        counter: Any,
        replay_history: list[Any],
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        pending_messages: Optional[list[Any]] = None,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> int:
        tool_payload = self.export_tools(tools)
        return self._get_codec().count_request_tokens(
            counter,
            replay_history,
            system_prompt=system_prompt,
            tools=tool_payload,
            pending_messages=pending_messages,
            reasoning=reasoning,
        )
    
    def _convert_messages(self, messages: list[dict[str, str] | Message]) -> list[Any]:
        """转换为当前 provider 的 request-ready replay 消息。"""
        return self._get_codec().prepare_messages(list(messages)) # type: ignore[arg-type]

    @staticmethod
    def _system_text_from_message(message: Any) -> Optional[str]:
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("thinking") or item.get("content")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str) and item:
                    parts.append(item)
            if parts:
                return "\n\n".join(parts)
        parts = message.get("parts")
        if isinstance(parts, list):
            texts = [
                str(part.get("text"))
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("text")
            ]
            if texts:
                return "\n\n".join(texts)
        return None

    def _prepare_request_input(
        self,
        messages: list[dict[str, str]] | ReplayRequestInput,
    ) -> ReplayRequestInput:
        if isinstance(messages, ReplayRequestInput):
            provider_name = getattr(self, "provider_name", None)
            if messages.provider_name and provider_name and messages.provider_name != provider_name:
                raise RuntimeError(
                    f"请求缓冲区属于 provider={messages.provider_name}，当前 LLM provider={provider_name}。"
                )
            return messages

        # 兼容模块
        prepared = self._convert_messages(messages) # type: ignore[arg-type]
        system_parts: list[str] = []
        replay_history: list[Any] = []
        for item in prepared:
            if isinstance(item, dict) and item.get("role") == "system":
                system_text = self._system_text_from_message(item)
                if system_text:
                    system_parts.append(system_text)
                continue
            replay_history.append(item)
        system_prompt = "\n\n".join(part for part in system_parts if part) or None
        return ReplayRequestInput(
            provider_name=getattr(self, "provider_name", None),
            replay_history=replay_history,
            system_prompt=system_prompt,
        )

    def prepare_messages_for_request(self, messages: list[Any]) -> list[Any]:
        """公开的 replay/history 预处理入口。"""
        return self._convert_messages(messages)

    def get_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容"""
        if response is None or isinstance(response, str):
            return None
        return self._get_codec().response_reasoning(response)

    def get_response_content(self, response: Any) -> Optional[str]:
        """提取响应文本内容（兼容 Chat API 和 Responses API）"""
        if response is None:
            return None
        if isinstance(response, str):
            return response
        return self._get_codec().response_text(response)

    def extract_usage_metrics(self, response: Any) -> dict[str, Any]:
        """Best-effort extraction of provider usage/cost metadata from a raw response."""
        if response is None or isinstance(response, str):
            return {}
        provider = getattr(self, "_provider", None)
        if provider is not None and hasattr(provider, "get_usage_from_response"):
            try:
                payload = provider.get_usage_from_response(response)
            except Exception:
                payload = {}
            if payload:
                return payload
        usage = _read_usage_value(response, "usage", "usage_metadata")
        if usage is None and hasattr(response, "model_dump"):
            try:
                payload = response.model_dump(mode="json")
            except Exception:
                payload = None
            if isinstance(payload, dict):
                usage = payload.get("usage") or payload.get("usage_metadata")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage") or response.get("usage_metadata")
        if usage is None:
            return {}

        input_tokens = _read_usage_value(
            usage,
            "prompt_tokens",
            "input_tokens",
            "inputTokens",
            "promptTokenCount",
            "inputTokenCount",
        )
        output_tokens = _read_usage_value(
            usage,
            "completion_tokens",
            "output_tokens",
            "outputTokens",
            "candidatesTokenCount",
            "outputTokenCount",
        )
        total_tokens = _read_usage_value(
            usage,
            "total_tokens",
            "totalTokens",
            "totalTokenCount",
        )
        reasoning_details = (
            _read_usage_value(usage, "output_token_details", "completion_tokens_details")
            or {}
        )
        reasoning_tokens = _read_usage_value(reasoning_details, "reasoning_tokens", "reasoningTokens")
        cached_input_tokens = _read_usage_value(
            _read_usage_value(usage, "input_token_details") or {},
            "cached_tokens",
            "cachedTokens",
        )
        cost_value = _read_usage_value(
            usage,
            "cost_usd",
            "costUsd",
            "total_cost",
            "totalCost",
            "total_cost_usd",
            "totalCostUsd",
        )

        try:
            input_tokens = int(input_tokens) if input_tokens is not None else None
        except Exception:
            input_tokens = None
        try:
            output_tokens = int(output_tokens) if output_tokens is not None else None
        except Exception:
            output_tokens = None
        try:
            total_tokens = int(total_tokens) if total_tokens is not None else None
        except Exception:
            total_tokens = None
        try:
            reasoning_tokens = int(reasoning_tokens) if reasoning_tokens is not None else None
        except Exception:
            reasoning_tokens = None
        try:
            cached_input_tokens = int(cached_input_tokens) if cached_input_tokens is not None else None
        except Exception:
            cached_input_tokens = None
        try:
            cost_value = float(cost_value) if cost_value is not None else None
        except Exception:
            cost_value = None

        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        payload = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens,
            "reasoningTokens": reasoning_tokens,
            "cachedInputTokens": cached_input_tokens,
            "costUsd": cost_value,
            "usageSource": "provider",
        }
        return {key: value for key, value in payload.items() if value is not None}

    def has_tool_calls(self, response: Any) -> bool:
        """检查是否有工具调用"""
        return self._get_codec().response_has_tool_calls(response)
    
    def get_tool_calls(self, response: Any) -> list:
        """获取工具调用列表"""
        return self._get_codec().response_tool_calls(response)
    
    def create_client(self):
        """创建客户端（向后兼容）"""
        return self._provider.client

    # ==================== 向后兼容属性 ====================

    @property
    def provide(self) -> str|None:
        """向后兼容：旧属性名 provide，请改用 provider_name"""
        return self.provider_name

    @provide.setter
    def provide(self, value: str | None) -> None:
        """向后兼容：允许旧测试/Mock 通过 provide 写入 provider_name。"""
        self.provider_name = value

    @property
    def resovle_api_key(self) -> str:
        """向后兼容：旧属性名 resovle_api_key，请改用 resolve_api_key"""
        return self.resolve_api_key

    @property
    def resovle_base_url(self) -> str:
        """向后兼容：旧属性名 resovle_base_url，请改用 resolve_base_url"""
        return self.resolve_base_url
