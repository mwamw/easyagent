"""
LLM Providers 模块

提供对不同 LLM 服务的统一适配。
"""
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .anthropic_provider import AnthropicProvider

__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "AnthropicProvider",
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
        provider_name: Provider 名称 (openai, google, anthropic, auto)
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
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
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
    else:
        return "openai"  # 默认使用 OpenAI 兼容
