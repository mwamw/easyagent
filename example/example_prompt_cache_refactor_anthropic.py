from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cache_policy import CacheableBlock, PromptCachePolicy
from core.history import _json_safe
from core.llm import EasyLLM
from core.request_input import ReplayRequestInput


ARTIFACT = PROJECT_ROOT / "example" / "_artifacts" / "prompt_cache_refactor_anthropic_request.json"


def main() -> None:
    llm = EasyLLM(
        provider="anthropic_native",
        base_url="http://127.0.0.1:5124",
        api_key="122",
        model="qwen3.5-9b",
    )

    request_input = ReplayRequestInput(
        provider_name=llm.provider_name,
        system_prompt_blocks=[
            CacheableBlock("identity", "你是一个简洁的中文助手。", partition="static"),
            CacheableBlock("policy", "回答要短，不要展开无关背景。", partition="session"),
        ],
        dynamic_context_blocks=[
            CacheableBlock("mailbox", "这是一条动态 mailbox，不应该进入 system cache prefix。", partition="dynamic", cacheable=False),
        ],
        cache_policy=PromptCachePolicy(enabled=True, ttl="1h"),
    )
    dynamic_context = request_input.render_dynamic_context()
    if dynamic_context:
        request_input.extend_replay(llm.query_to_replay(dynamic_context))
    request_input.extend_replay(llm.query_to_replay("用一句话说明 prompt cache refactor 的目的。"))

    captured: dict[str, Any] = {}
    original_apply = llm._provider.apply_cache_policy

    def capture_apply(request: Any, prepared_input: ReplayRequestInput) -> Any:
        updated = original_apply(request, prepared_input)
        captured["request"] = _json_safe(updated)
        captured["cache_metadata"] = _json_safe(prepared_input.cache_metadata)
        return updated

    llm._provider.apply_cache_policy = capture_apply  # type: ignore[method-assign]
    response = llm.invoke_raw(request_input)

    payload = {
        "captured": captured,
        "response": _json_safe(response),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(ARTIFACT), "cacheMetadata": captured.get("cache_metadata")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
