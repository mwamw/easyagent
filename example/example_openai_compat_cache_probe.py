from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT_ROOT / "example"
ARTIFACT = EXAMPLE_DIR / "_artifacts" / "openai_compat_cache_probe.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm import EasyLLM


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_cached_tokens(raw_usage: Any, normalized_usage: dict[str, Any]) -> int | None:
    cached = normalized_usage.get("cachedInputTokens")
    if isinstance(cached, (int, float)):
        return int(cached)
    if isinstance(raw_usage, dict):
        prompt_details = raw_usage.get("prompt_tokens_details") or raw_usage.get("input_tokens_details") or {}
        cached = prompt_details.get("cached_tokens")
        if isinstance(cached, (int, float)):
            return int(cached)
        cached = raw_usage.get("cached_content_token_count")
        if isinstance(cached, (int, float)):
            return int(cached)
    return None


def _build_messages() -> list[dict[str, str]]:
    stable_prefix = (
        "你是一个严谨的中文技术助手。"
        "请严格遵循这些要求："
        "回答先给结论，再给理由；"
        "不要输出无关背景；"
        "若信息不足，明确说信息不足。"
    ) * 40
    return [
        {"role": "system", "content": stable_prefix},
        {"role": "user", "content": "请用一句话解释什么是 prompt cache。"},
    ]


def _build_llm() -> EasyLLM:
    load_dotenv(EXAMPLE_DIR / ".env", override=True)
    base_url = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("LLM_BASE_URL 未配置，无法构造 openai 兼容接口地址。")
    base_url = f"{base_url}/v1"
    return EasyLLM(
        provider="openai",
        base_url=base_url,
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL_ID"),
    )


def run_probe(rounds: int = 3) -> dict[str, Any]:
    llm = _build_llm()
    messages = _build_messages()
    runs: list[dict[str, Any]] = []

    for index in range(rounds):
        entry: dict[str, Any] = {
            "round": index + 1,
            "provider": llm.provider_name,
            "model": llm.model,
            "base_url": llm.base_url,
        }
        try:
            response = llm.invoke_raw(messages)
            normalized_usage = llm.extract_usage_metrics(response)
            raw_usage = getattr(response, "usage", None)
            if raw_usage is None and isinstance(response, dict):
                raw_usage = response.get("usage") or response.get("usage_metadata")
            raw_usage = _json_safe(raw_usage)
            entry["status"] = "ok"
            entry["normalized_usage"] = normalized_usage
            entry["raw_usage"] = raw_usage
            entry["cached_tokens"] = _extract_cached_tokens(raw_usage, normalized_usage)
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message
                entry["content_preview"] = getattr(message, "content", None)
            elif isinstance(response, dict):
                entry["content_preview"] = response.get("content")
        except Exception as exc:
            entry["status"] = "error"
            entry["error_type"] = type(exc).__name__
            entry["error"] = str(exc)
        runs.append(entry)

    result = {
        "summary": {
            "provider": llm.provider_name,
            "model": llm.model,
            "base_url": llm.base_url,
            "rounds": rounds,
            "supports_cache_usage_field": any(run.get("cached_tokens") is not None for run in runs),
        },
        "runs": runs,
    }
    return result


def main() -> None:
    result = run_probe(rounds=3)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== OpenAI Compat Cache Probe ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for run in result["runs"]:
        print(f"\n--- Round {run['round']} ---")
        print(json.dumps(run, ensure_ascii=False, indent=2))
    print(f"\nartifact: {ARTIFACT}")


if __name__ == "__main__":
    main()
