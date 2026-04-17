"""Provider and codec factory exports."""

from __future__ import annotations

from typing import Optional

from .anthropic_compat import AnthropicCompatCodec, AnthropicProvider
from .anthropic_native import AnthropicNativeCodec, AnthropicNativeProvider
from .base import BaseProvider
from .google_compat import GoogleCompatCodec, GoogleProvider
from .google_native import GoogleNativeCodec, GoogleNativeProvider
from .openai_compat import OpenAIChatCodec, OpenAICompatibleProviderBase, OpenAIProvider
from .openai_responses import OpenAIResponsesCodec, OpenAIResponsesProvider
from .shared import BaseProviderCodec

__all__ = [
    "BaseProvider",
    "BaseProviderCodec",
    "OpenAICompatibleProviderBase",
    "OpenAIProvider",
    "OpenAIResponsesProvider",
    "GoogleProvider",
    "AnthropicProvider",
    "GoogleNativeProvider",
    "AnthropicNativeProvider",
    "OpenAIChatCodec",
    "GoogleCompatCodec",
    "AnthropicCompatCodec",
    "OpenAIResponsesCodec",
    "GoogleNativeCodec",
    "AnthropicNativeCodec",
    "create_codec",
    "create_provider",
    "provider_requires_base_url",
    "detect_provider_from_model",
]


def create_provider(
    provider_name: str,
    model: str,
    api_key: str,
    base_url: str,
    **kwargs
) -> BaseProvider:
    """
    工厂函数：根据 provider 名称创建对应的 Provider 实例
    
    Args:
        provider_name: Provider 名称 (openai, google, google_native, anthropic, anthropic_native, auto, openai_responses)
        model: 模型名称
        api_key: API 密钥
        base_url: API 地址
        **kwargs: 其他参数
        
    Returns:
        Provider 实例
    """
    provider_name = provider_name.lower()
    
    # 如果是 auto，根据模型名推断
    if provider_name == "auto":
        provider_name = detect_provider_from_model(model)
    
    provider_map = {
        "openai": OpenAIProvider,
        "openai_responses": OpenAIResponsesProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "google_native": GoogleNativeProvider,
        "gemini_native": GoogleNativeProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "anthropic_native": AnthropicNativeProvider,
        "claude_native": AnthropicNativeProvider,
        # 以下都使用 OpenAI 兼容接口
        "deepseek": OpenAIProvider,
        "qwen": OpenAIProvider,
        "kimi": OpenAIProvider,
        "moonshot": OpenAIProvider,
        "zhipu": OpenAIProvider,
        "glm": OpenAIProvider,
        "ollama": OpenAIProvider,
        "vllm": OpenAIProvider,
        "modelscope": OpenAIProvider,
    }
    
    provider_class = provider_map.get(provider_name, OpenAIProvider)
    return provider_class(model=model, api_key=api_key, base_url=base_url, **kwargs)


def create_codec(provider_name: Optional[str]) -> BaseProviderCodec:
    normalized = (provider_name or "openai").lower()
    if normalized in {"google_native", "gemini_native"}:
        return GoogleNativeCodec(normalized)
    if normalized in {"anthropic_native", "claude_native"}:
        return AnthropicNativeCodec(normalized)
    if normalized == "openai_responses":
        return OpenAIResponsesCodec(normalized)
    if normalized in {"google", "gemini"}:
        return GoogleCompatCodec(normalized)
    if normalized in {"anthropic", "claude"}:
        return AnthropicCompatCodec(normalized)
    return OpenAIChatCodec(normalized)


def provider_requires_base_url(provider_name: str) -> bool:
    """Return whether the provider requires an explicit base_url to function."""
    normalized = (provider_name or "").lower()
    return normalized not in {"google_native", "gemini_native", "anthropic_native", "claude_native"}


def detect_provider_from_model(model: str) -> str:
    """
    根据模型名称推断 Provider
    
    Args:
        model: 模型名称
        
    Returns:
        Provider 名称
    """
    if not model:
        return "openai"
    
    model_lower = model.lower()
    
    # 检测模型类型
    if "gpt" in model_lower:
        return "openai"
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
    else:
        return "openai"  # 默认使用 OpenAI 兼容
