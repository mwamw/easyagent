from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT_ROOT / "example"
ARTIFACT = EXAMPLE_DIR / "_artifacts" / "agent_cache_realworld_probe.json"
REAL_SKILLS_DIR = PROJECT_ROOT / "test" / "real_skills"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from core.llm import EasyLLM
from skill.builtin.calculator_skill import CalculatorSkill
from skill.meta_tools import MetaSkill
from skill.registry import SkillRegistry


MODEL = "deepseek-v4-flash:zenmux:claude"
RUN_SALT = f"agent-cache-probe-{int(time.time())}"
STABLE_SYSTEM_PROMPT = (
    "你是一个严谨的中文技术助手。"
    "回答必须先给结论再给理由；"
    "不要输出无关背景；"
    "若信息不足就明确说信息不足；"
    "默认使用短句；"
    "若涉及工具，不要假装调用。"
) * 60


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


def _build_llm() -> EasyLLM:
    load_dotenv(EXAMPLE_DIR / ".env", override=True)
    base_url = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY")
    if not base_url:
        raise RuntimeError("LLM_BASE_URL 未配置。")
    return EasyLLM(
        provider="anthropic_native",
        base_url=base_url,
        api_key=api_key,
        model=MODEL,
    )


def _event_time(event: dict[str, Any]) -> str:
    return str(event.get("endedAt") or event.get("startedAt") or event.get("createdAt") or "")


def _normalize_llm_event(event: dict[str, Any]) -> dict[str, Any]:
    input_tokens = int(event.get("inputTokens") or 0)
    cache_read_tokens_raw = event.get("cacheReadTokens")
    cache_read_tokens = int(cache_read_tokens_raw or 0)
    cached_input_tokens = int(event.get("cachedInputTokens") or 0)
    provider_name = str(event.get("providerName") or "")

    if cache_read_tokens_raw is not None or provider_name == "anthropic_native":
        total_prompt_tokens = input_tokens + cache_read_tokens
        cache_hit_tokens = cache_read_tokens
        uncached_input_tokens = input_tokens
        semantic = "anthropic_style"
    else:
        total_prompt_tokens = input_tokens
        cache_hit_tokens = min(cached_input_tokens, total_prompt_tokens)
        uncached_input_tokens = max(0, total_prompt_tokens - cache_hit_tokens)
        semantic = "openai_style"

    cache_hit_ratio = (
        float(cache_hit_tokens) / float(total_prompt_tokens)
        if total_prompt_tokens > 0
        else None
    )
    return {
        "id": event.get("id"),
        "requestKind": event.get("requestKind"),
        "providerName": provider_name,
        "model": event.get("model"),
        "toolsEnabled": bool(event.get("toolsEnabled")),
        "inputTokens": input_tokens,
        "outputTokens": int(event.get("outputTokens") or 0),
        "totalTokens": int(event.get("totalTokens") or 0),
        "cachedInputTokens": cached_input_tokens,
        "cacheReadTokens": cache_read_tokens,
        "cacheCreationTokens": int(event.get("cacheCreationTokens") or 0),
        "reasoningTokens": int(event.get("reasoningTokens") or 0),
        "semantic": semantic,
        "totalPromptTokens": total_prompt_tokens,
        "uncachedInputTokens": uncached_input_tokens,
        "cacheHitTokens": cache_hit_tokens,
        "cacheHitRatio": cache_hit_ratio,
        "usageSource": event.get("usageSource"),
        "metadata": _json_safe(event.get("metadata") or {}),
        "endedAt": event.get("endedAt"),
    }


def _summarize_llm_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [_normalize_llm_event(event) for event in events]
    total_prompt_tokens = sum(int(item["totalPromptTokens"]) for item in normalized)
    cache_hit_tokens = sum(int(item["cacheHitTokens"]) for item in normalized)
    uncached_input_tokens = sum(int(item["uncachedInputTokens"]) for item in normalized)
    return {
        "llmRequestCount": len(normalized),
        "totalPromptTokens": total_prompt_tokens,
        "cacheHitTokens": cache_hit_tokens,
        "uncachedInputTokens": uncached_input_tokens,
        "cacheHitRatio": (
            float(cache_hit_tokens) / float(total_prompt_tokens)
            if total_prompt_tokens > 0
            else None
        ),
        "requests": normalized,
    }


def _collect_new_events(
    agent: BasicAgent,
    *,
    event_type: str,
    seen_ids: set[str],
    limit: int = 500,
) -> list[dict[str, Any]]:
    events = agent.get_recent_observability_events(limit=limit, event_type=event_type)
    new_events = [dict(event) for event in events if str(event.get("id")) not in seen_ids]
    for event in new_events:
        seen_ids.add(str(event.get("id")))
    new_events.sort(key=_event_time)
    return new_events


def _invoke_and_collect(
    agent: BasicAgent,
    *,
    query: str,
    seen_llm_ids: set[str],
    seen_break_ids: set[str],
    note: str,
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            response = agent.invoke(query)
            break
        except Exception as exc:
            if "429" not in str(exc) or attempt >= 2:
                raise
            # 丢弃本次失败尝试产生的 telemetry，避免污染最终统计。
            _collect_new_events(agent, event_type="llm", seen_ids=seen_llm_ids)
            _collect_new_events(agent, event_type="cache_break", seen_ids=seen_break_ids)
            time.sleep(6 * (attempt + 1))
            attempt += 1
    llm_events = _collect_new_events(agent, event_type="llm", seen_ids=seen_llm_ids)
    cache_breaks = _collect_new_events(agent, event_type="cache_break", seen_ids=seen_break_ids)
    context_usage = agent.get_context_usage()
    time.sleep(2)
    return {
        "note": note,
        "query": query,
        "responsePreview": str(response)[:400],
        "llm": _summarize_llm_events(llm_events),
        "cacheBreaks": [
            {
                "reason": item.get("reason"),
                "changedFields": item.get("changedFields"),
                "previousCacheReadTokens": item.get("previousCacheReadTokens"),
                "currentCacheReadTokens": item.get("currentCacheReadTokens"),
                "metadata": _json_safe(item.get("metadata") or {}),
            }
            for item in cache_breaks
        ],
        "contextUsage": _json_safe(context_usage),
    }


def _new_agent(
    *,
    llm: EasyLLM | None = None,
    name: str,
    enable_tool: bool = False,
    scenario_tag: str,
) -> BasicAgent:
    return BasicAgent(
        name=name,
        llm=llm or _build_llm(),
        system_prompt=f"{STABLE_SYSTEM_PROMPT}\n\n## Cache Probe Tag\n{RUN_SALT}:{scenario_tag}",
        enable_tool=enable_tool,
        tool_registry=ToolRegistry() if enable_tool else None,
        verbose_thinking=False,
    )


def scenario_stateless_repeated() -> dict[str, Any]:
    agent = _new_agent(name="cache-stateless", enable_tool=False, scenario_tag="stateless")
    seen_llm_ids: set[str] = set()
    seen_break_ids: set[str] = set()
    query = "请用一句话解释什么是 prompt cache。"
    invokes = []
    for round_index in range(3):
        agent.clear_history()
        invokes.append(
            _invoke_and_collect(
                agent,
                query=query,
                seen_llm_ids=seen_llm_ids,
                seen_break_ids=seen_break_ids,
                note=f"stateless_round_{round_index + 1}",
            )
        )
    return {
        "scenario": "stateless_repeated",
        "description": "每次 invoke 前清空 history，观察稳定 system prompt 下的 provider prompt cache 命中。",
        "invokes": invokes,
        "summary": _json_safe(agent.get_observability_summary()),
    }


def scenario_growing_history() -> dict[str, Any]:
    agent = _new_agent(name="cache-history", enable_tool=False, scenario_tag="history")
    seen_llm_ids: set[str] = set()
    seen_break_ids: set[str] = set()
    queries = [
        "请用一句话解释什么是 prompt cache。",
        "再用一句话说明它为什么能降低延迟。",
        "再用一句话说明命中 cache 的关键前提是什么。",
    ]
    invokes = []
    for index, query in enumerate(queries, start=1):
        invokes.append(
            _invoke_and_collect(
                agent,
                query=query,
                seen_llm_ids=seen_llm_ids,
                seen_break_ids=seen_break_ids,
                note=f"history_round_{index}",
            )
        )
    return {
        "scenario": "growing_history",
        "description": "保留完整对话历史，观察 history 增长时的 cache 命中比例变化。",
        "invokes": invokes,
        "summary": _json_safe(agent.get_observability_summary()),
    }


def scenario_resident_skill_toggle() -> dict[str, Any]:
    agent = _new_agent(name="cache-resident-skill", enable_tool=True, scenario_tag="resident-skill")
    seen_llm_ids: set[str] = set()
    seen_break_ids: set[str] = set()
    query = "请只用一句话回答：什么是 prompt cache？不要调用任何工具。"

    invokes = []
    invokes.append(
        _invoke_and_collect(
            agent,
            query=query,
            seen_llm_ids=seen_llm_ids,
            seen_break_ids=seen_break_ids,
            note="before_skill",
        )
    )

    agent.with_skill(CalculatorSkill())
    invokes.append(
        _invoke_and_collect(
            agent,
            query=query,
            seen_llm_ids=seen_llm_ids,
            seen_break_ids=seen_break_ids,
            note="after_skill_activation",
        )
    )
    invokes.append(
        _invoke_and_collect(
            agent,
            query=query,
            seen_llm_ids=seen_llm_ids,
            seen_break_ids=seen_break_ids,
            note="skill_warm_repeat",
        )
    )

    agent.remove_skill("calculator")
    invokes.append(
        _invoke_and_collect(
            agent,
            query=query,
            seen_llm_ids=seen_llm_ids,
            seen_break_ids=seen_break_ids,
            note="after_skill_removal",
        )
    )
    return {
        "scenario": "resident_skill_toggle",
        "description": "常驻 skill 改变 system/tools 签名，再移除 skill 观察是否恢复旧前缀 cache。",
        "invokes": invokes,
        "summary": _json_safe(agent.get_observability_summary()),
    }


def scenario_runtime_skill_ephemeral() -> dict[str, Any]:
    SkillRegistry.reset()
    registry = SkillRegistry.instance()
    registry.discover_from_directory(str(REAL_SKILLS_DIR))

    agent = _new_agent(name="cache-runtime-skill", enable_tool=True, scenario_tag="runtime-skill")
    agent.with_skill(MetaSkill(registry, agent.skill_manager))

    seen_llm_ids: set[str] = set()
    seen_break_ids: set[str] = set()
    query = (
        "必须真实使用工具，并严格按流程执行："
        "1. 先调用 skill_tool，skill_name=crypto_skill；"
        "2. 再调用该 skill 暴露的 hash_calculator，计算字符串 Autonomous Mode 的 SHA-256；"
        "3. 最后只输出最终哈希值。"
    )

    invokes = []
    for round_index in range(2):
        result = _invoke_and_collect(
            agent,
            query=query,
            seen_llm_ids=seen_llm_ids,
            seen_break_ids=seen_break_ids,
            note=f"runtime_skill_round_{round_index + 1}",
        )
        result["postInvokeState"] = {
            "hasRuntimeSkillContext": agent.skill_manager.has_runtime_skill_context(),
            "cryptoSkillRegistered": agent.skill_manager.has_skill("crypto_skill"),
            "cryptoSkillActive": agent.skill_manager.is_active("crypto_skill") if agent.skill_manager.has_skill("crypto_skill") else False,
            "activeSkills": list(agent.skill_manager.active_skill_names),
        }
        invokes.append(result)

    summary = _json_safe(agent.get_observability_summary())
    SkillRegistry.reset()
    return {
        "scenario": "runtime_skill_ephemeral",
        "description": "按需 skill_tool 注入 crypto_skill 正文和工具，再检查 invoke 结束后临时 skill/context 是否清理，以及下一轮 cache 命中是否恢复稳定前缀。",
        "invokes": invokes,
        "summary": summary,
    }


def main() -> None:
    results = {
        "provider": "anthropic_native",
        "model": MODEL,
        "base_url": (os.getenv("LLM_BASE_URL") or "").rstrip("/"),
        "scenarios": [
            scenario_stateless_repeated(),
            scenario_growing_history(),
            scenario_resident_skill_toggle(),
            scenario_runtime_skill_ephemeral(),
        ],
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"artifact": str(ARTIFACT), "model": MODEL, "scenarioCount": len(results["scenarios"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
