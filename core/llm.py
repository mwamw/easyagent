"""
EasyLLM - 统一的 LLM 接口

提供对不同 LLM 服务的统一访问接口。
"""
from .Message import Message
from .providers import create_provider, BaseProvider, provider_requires_base_url
from typing import Optional, Any, Generator, AsyncGenerator
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
        
        # 保持向后兼容
        self.client = self._provider.client
        
        logger.info(f"EasyLLM 初始化完成: provider={self.provider_name}, model={self.model}")
    
    @property
    def provider(self) -> BaseProvider:
        """获取当前 Provider"""
        return self._provider
    
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
    
    def invoke(
        self,
        messages: list[dict[str, str] | Message],
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
        messages = self._convert_messages(messages) # type: ignore
        return self._provider.invoke(messages, temperature=temperature,reasoning=reasoning, **kwargs) # type: ignore

    def invoke_raw(
        self,
        messages: list[dict[str, str] | Message],
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any:
        """同步调用并返回 provider 原始响应对象。"""
        messages = self._convert_messages(messages) # type: ignore
        provider = getattr(self, "_provider", None)
        if provider is not None:
            if hasattr(provider, "invoke_raw"):
                return provider.invoke_raw(messages, temperature=temperature, reasoning=reasoning, **kwargs)  # type: ignore
            return provider.invoke(messages, temperature=temperature, reasoning=reasoning, **kwargs)  # type: ignore
        return self.invoke(messages, temperature=temperature, reasoning=reasoning, **kwargs)
    
    def stream(
        self,
        messages: list[dict[str, str] | Message],
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
        messages = self._convert_messages(messages) # type: ignore
        yield from self._provider.stream(messages, temperature=temperature,reasoning=reasoning, **kwargs)

    def stream_events(
        self,
        messages: list[dict[str, str] | Message],
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """流式调用，返回 thinking / text / final 事件。"""
        messages = self._convert_messages(messages) # type: ignore
        provider = getattr(self, "_provider", None)
        if provider is not None and hasattr(provider, "stream_events"):
            yield from provider.stream_events(messages, temperature=temperature, reasoning=reasoning, **kwargs)  # type: ignore
            return

        collected = []
        stream_source = provider.stream if provider is not None else self.stream  # type: ignore[attr-defined]
        for chunk in stream_source(messages, temperature=temperature, reasoning=reasoning, **kwargs):  # type: ignore
            collected.append(chunk)
            yield {"type": "text_delta", "delta": chunk}
        yield {
            "type": "final_response",
            "content": "".join(collected),
            "thinking": "",
        }
    
    def invoke_with_tools(
        self,
        messages: list[dict[str, str] | Message],
        tools: list[dict],
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
        messages = self._convert_messages(messages) # type: ignore
        try:
            result=self._provider.invoke_with_tools(messages, tools,reasoning=reasoning, temperature=temperature, **kwargs)
            return result
        except Exception as e:
            logger.error(f"LLM工具调用失败 当前消息{messages[-1]}")
            raise e

    def stream_with_tools(
        self,
        messages: list[dict[str, str] | Message],
        tools: list[dict],
        reasoning: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """同步流式带工具调用，返回统一事件流。"""
        messages = self._convert_messages(messages) # type: ignore
        yield from self._provider.stream_with_tools(
            messages,
            tools,
            reasoning=reasoning,
            temperature=temperature,
            **kwargs,
        )

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
    
    def format_assistant_response(self, response: Any, include_reasoning: bool = False) -> Any:
        """
        格式化 assistant 响应为当前 Provider 需要的格式
        
        用于处理 tool_calls 消息，不同的 Provider 有不同的格式要求。
        
        Args:
            response: LLM 响应对象
            
        Returns:
            格式化后的消息字典
        """
        if isinstance(response, str):
            return self.format_assistant_message(content=response)
        provider = getattr(self, "_provider", None)
        # 检查 Provider 是否有这个方法
        if provider is not None and hasattr(provider, 'format_assistant_response'):
            try:
                return provider.format_assistant_response(response, include_reasoning=include_reasoning) #type: ignore
            except TypeError:
                return provider.format_assistant_response(response) #type: ignore
        
        # 默认：直接返回原始响应（OpenAI 兼容格式可以直接使用）
        return response

    def format_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        thinking: Optional[str] = None,
    ) -> Any:
        """基于统一 tool call 结构构造 assistant 消息。"""
        provider = getattr(self, "_provider", None)
        if provider is not None and hasattr(provider, "format_assistant_message"):
            try:
                return provider.format_assistant_message( # type: ignore
                    content=content,
                    tool_calls=tool_calls,
                    thinking=thinking,
                )
            except TypeError:
                return provider.format_assistant_message( # type: ignore
                    content=content,
                    tool_calls=tool_calls,
                )
        return {
            "role": "assistant",
            "content": content or "",
            **({"reasoning_content": thinking} if thinking else {}),
        }
    
    # ==================== 异步 API ====================

    async def ainvoke(
        self,
        messages: list[dict[str, str] | Message],
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
        messages = self._convert_messages(messages) # type: ignore
        return await self._provider.async_invoke(messages, temperature=temperature,reasoning=reasoning, **kwargs) # type: ignore

    async def ainvoke_raw(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """异步调用并返回 provider 原始响应对象。"""
        messages = self._convert_messages(messages) # type: ignore
        provider = getattr(self, "_provider", None)
        if provider is not None:
            if hasattr(provider, "async_invoke_raw"):
                return await provider.async_invoke_raw(messages, temperature=temperature, reasoning=reasoning, **kwargs)  # type: ignore
            return await provider.async_invoke(messages, temperature=temperature, reasoning=reasoning, **kwargs)  # type: ignore
        return await self.ainvoke(messages, temperature=temperature, reasoning=reasoning, **kwargs)

    async def astream(
        self,
        messages: list[dict[str, str] | Message],
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
        messages = self._convert_messages(messages) # type: ignore
        async for chunk in self._provider.async_stream(messages, reasoning=reasoning,temperature=temperature, **kwargs):
            yield chunk

    async def astream_events(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式调用，返回 thinking / text / final 事件。"""
        messages = self._convert_messages(messages) # type: ignore
        provider = getattr(self, "_provider", None)
        if provider is not None and hasattr(provider, "async_stream_events"):
            async for event in provider.async_stream_events(messages, reasoning=reasoning, temperature=temperature, **kwargs):  # type: ignore
                yield event
            return

        collected = []
        stream_source = provider.async_stream if provider is not None else self.astream  # type: ignore[attr-defined]
        async for chunk in stream_source(messages, reasoning=reasoning, temperature=temperature, **kwargs):  # type: ignore
            collected.append(chunk)
            yield {"type": "text_delta", "delta": chunk}
        yield {
            "type": "final_response",
            "content": "".join(collected),
            "thinking": "",
        }

    async def ainvoke_with_tools(
        self,
        messages: list[dict[str, str] | Message],
        tools: list[dict],
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
        messages = self._convert_messages(messages) # type: ignore
        try:
            result = await self._provider.async_invoke_with_tools(messages, tools,reasoning=reasoning, temperature=temperature, **kwargs)
            return result
        except Exception as e:
            logger.error(f"LLM 异步工具调用失败 当前消息{messages[-1]}")
            raise e

    async def astream_with_tools(
        self,
        messages: list[dict[str, str] | Message],
        tools: list[dict],
        temperature: Optional[float] = None,
        reasoning: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """异步流式带工具调用，返回统一事件流。"""
        messages = self._convert_messages(messages) # type: ignore
        async for event in self._provider.async_stream_with_tools(
            messages,
            tools,
            reasoning=reasoning,
            temperature=temperature,
            **kwargs,
        ):
            yield event

    # ==================== 向后兼容的方法 ====================
    
    def think(
        self,
        messages: list[dict[str, str] | Message],
        temperature: Optional[float] = None ,
        reasoning: Optional[dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """流式输出（向后兼容）"""
        messages = self._convert_messages(messages) # type: ignore
        for chunk in self._provider.stream(messages, temperature=temperature,reasoning=reasoning):
            print(chunk, end="", flush=True)
            yield chunk
    
    def stream_invoke(
        self,
        messages: list[dict[str, str] | Message],
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
    
    def _convert_messages(self, messages: list[dict[str, str] | Message]) -> list[dict[str, str]]:
        """将 Message 对象转换为字典"""
        provider = getattr(self, "_provider", None)
        payloads = [msg.to_dict() if isinstance(msg, Message) else msg for msg in messages]
        if provider is not None and hasattr(provider, "prepare_messages_for_request"):
            return provider.prepare_messages_for_request(payloads)  # type: ignore[return-value]

        converted = []
        for payload in payloads:
            if provider is not None and hasattr(provider, "prepare_message_for_request"):
                payload = provider.prepare_message_for_request(payload)  # type: ignore
            if payload is None:
                continue
            converted.append(payload)
        return converted

    def get_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容"""
        if response is None or isinstance(response, str):
            return None
        provider = getattr(self, "_provider", None)
        if provider is None:
            return getattr(response, "reasoning_content", None)
        return self._provider.get_thinking_content(response)

    def get_response_content(self, response: Any) -> Optional[str]:
        """提取响应文本内容（兼容 Chat API 和 Responses API）"""
        if response is None:
            return None
        if isinstance(response, str):
            return response
        provider = getattr(self, "_provider", None)
        if provider is None:
            return getattr(response, "content", None)
        return self._provider.get_response_content(response)

    def has_tool_calls(self, response: Any) -> bool:
        """检查是否有工具调用"""
        provider = getattr(self, "_provider", None)
        if provider is None:
            return bool(getattr(response, "tool_calls", None))
        return self._provider.has_tool_calls(response)
    
    def get_tool_calls(self, response: Any) -> list:
        """获取工具调用列表"""
        provider = getattr(self, "_provider", None)
        if provider is None:
            return getattr(response, "tool_calls", []) or []
        return self._provider.get_tool_calls(response)
    
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
