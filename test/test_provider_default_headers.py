from unittest.mock import MagicMock, patch

from core.llm import EasyLLM
from core.providers import AnthropicNativeProvider, OpenAIProvider


def test_easyllm_forwards_default_headers_to_provider() -> None:
    provider = MagicMock()
    provider.client = object()

    with patch("core.llm.create_provider", return_value=provider) as create_provider:
        EasyLLM(
            provider="openai",
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
            default_headers={"User-Agent": "CustomAgent/1.0"},
        )

    assert create_provider.call_args.kwargs["default_headers"] == {
        "User-Agent": "CustomAgent/1.0"
    }


def test_openai_provider_applies_default_headers_to_sync_and_async_clients() -> None:
    headers = {"User-Agent": "S4Code/0.1.0"}

    with (
        patch("core.providers.openai_compat.transport.OpenAI") as sync_client,
        patch("core.providers.openai_compat.transport.AsyncOpenAI") as async_client,
    ):
        provider = OpenAIProvider(
            model="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
            default_headers=headers,
        )
        provider._get_async_client()

    assert sync_client.call_args.kwargs["default_headers"] == headers
    assert async_client.call_args.kwargs["default_headers"] == headers


def test_anthropic_provider_applies_default_headers_to_sync_and_async_clients() -> None:
    headers = {"User-Agent": "S4Code/0.1.0"}

    with (
        patch("anthropic.Anthropic") as sync_client,
        patch("anthropic.AsyncAnthropic") as async_client,
    ):
        provider = AnthropicNativeProvider(
            model="test-model",
            api_key="test-key",
            base_url="https://example.test",
            default_headers=headers,
        )
        provider._get_async_client()

    assert sync_client.call_args.kwargs["default_headers"] == headers
    assert async_client.call_args.kwargs["default_headers"] == headers
