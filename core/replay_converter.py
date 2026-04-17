from __future__ import annotations

import json
from typing import Any, Optional

from .history import CanonicalMessage, _reasoning_text, coerce_canonical_message


def _normalize_provider_name(provider_name: Optional[str]) -> str:
    normalized = (provider_name or "openai").lower()
    aliases = {
        "gemini": "google",
        "gemini_native": "google_native",
        "claude": "anthropic",
        "claude_native": "anthropic_native",
        "moonshot": "kimi",
        "glm": "zhipu",
    }
    return aliases.get(normalized, normalized)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _reasoning_from_block(block: Any) -> str:
    if getattr(block, "text", None):
        return str(getattr(block, "text"))
    if getattr(block, "summary", None) is not None:
        return _reasoning_text(getattr(block, "summary"))
    payload = getattr(block, "payload", None)
    if payload is not None:
        return _reasoning_text(payload)
    return ""


def _json_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False, default=str)


def _dict_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            loaded = json.loads(arguments)
        except Exception:
            return {}
        if isinstance(loaded, dict):
            return loaded
        return {}
    return {}


def _function_response_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return dict(output)
    return {"result": output}


def _canonical_to_openai_like_message(
    canonical: CanonicalMessage,
    *,
    tool_role: str = "tool",
    preserve_reasoning: bool = True,
) -> list[Any]:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for block in canonical.content:
        if block.type == "text" and block.text:
            text_parts.append(block.text)
            continue
        if block.type == "reasoning":
            text = _reasoning_from_block(block)
            if text:
                thinking_parts.append(text)
            continue
        if block.type == "function_call":
            payload = block.payload if isinstance(block.payload, dict) else None
            if isinstance(payload, dict) and payload.get("type") == "function":
                tool_calls.append(dict(payload))
                continue
            tool_calls.append(
                {
                    "id": block.call_id,
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": _json_arguments(block.arguments),
                    },
                }
            )
            continue
        if block.type == "function_response":
            tool_results.append(
                {
                    "role": tool_role,
                    "content": _stringify(block.output),
                    "tool_call_id": block.call_id or "",
                    "name": block.name,
                }
            )

    if canonical.role == "assistant":
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            payload["tool_calls"] = tool_calls
        if preserve_reasoning and thinking_parts:
            payload["reasoning_content"] = "".join(thinking_parts)
        return [payload]

    if canonical.role == "tool":
        if tool_results:
            return tool_results
        return [{"role": tool_role, "content": "", "tool_call_id": "", "name": None}]

    payload = {
        "role": canonical.role,
        "content": "".join(text_parts),
    }
    if preserve_reasoning and thinking_parts:
        payload["reasoning_content"] = "".join(thinking_parts)
    return [payload]


def _canonical_to_anthropic_message(
    canonical: CanonicalMessage,
    *,
    include_reasoning_sidecar: bool,
) -> list[Any]:
    blocks: list[dict[str, Any]] = []
    thinking_parts: list[str] = []

    for block in canonical.content:
        payload = block.payload if isinstance(block.payload, dict) else None
        provider_block_type = block.metadata.get("provider_block_type") if isinstance(block.metadata, dict) else None

        if block.type == "text":
            if block.text:
                blocks.append({"type": "text", "text": block.text})
            continue
        if block.type == "reasoning":
            text = _reasoning_from_block(block)
            if text:
                thinking_parts.append(text)
            if isinstance(payload, dict) and payload.get("type") in {"thinking", "redacted_thinking"}:
                blocks.append(dict(payload))
                continue
            if provider_block_type == "redacted_thinking" and isinstance(payload, dict):
                blocks.append(dict(payload))
                continue
            thinking_block: dict[str, Any] = {
                "type": "thinking",
                "thinking": text,
            }
            if block.signature:
                thinking_block["signature"] = block.signature
            blocks.append(thinking_block)
            continue
        if block.type == "function_call":
            if isinstance(payload, dict) and payload.get("type") == "tool_use":
                blocks.append(dict(payload))
                continue
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.call_id,
                    "name": block.name or "",
                    "input": _dict_arguments(block.arguments),
                }
            )
            continue
        if block.type == "function_response":
            content = block.output
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            tool_result = {
                "type": "tool_result",
                "tool_use_id": block.call_id,
                "content": content,
            }
            if block.name:
                tool_result["name"] = block.name
            blocks.append(tool_result)
            continue
        if block.type == "provider_item" and isinstance(payload, dict):
            blocks.append(dict(payload))

    if canonical.role == "tool":
        return [{"role": "user", "content": blocks or [{"type": "tool_result", "tool_use_id": None, "content": ""}]}]

    if canonical.role == "system":
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict) and block.get("type") == "text")
        if not text:
            text = canonical.text_content()
        return [{"role": "system", "content": text}]

    content: Any
    if blocks and all(isinstance(block, dict) and block.get("type") == "text" for block in blocks):
        content = "".join(block.get("text", "") for block in blocks)
    else:
        content = blocks

    payload: dict[str, Any] = {
        "role": "assistant" if canonical.role == "assistant" else "user",
        "content": content,
    }
    if include_reasoning_sidecar and thinking_parts:
        payload["reasoning_content"] = "".join(thinking_parts)
    return [payload]


def _google_part_from_provider_payload(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    if "text" in payload or "function_call" in payload or "function_response" in payload or "inline_data" in payload:
        return dict(payload)
    block_type = payload.get("type")
    if block_type == "text":
        return {"text": payload.get("text", "")}
    if block_type == "thinking":
        part = {"text": payload.get("text") or payload.get("thinking", ""), "thought": True}
        signature = payload.get("thought_signature") or payload.get("thoughtSignature")
        if signature is not None:
            part["thought_signature"] = signature
        return part
    if block_type == "function_call":
        part = {
            "function_call": {
                "id": payload.get("id"),
                "name": payload.get("name", ""),
                "args": payload.get("args", {}) or {},
            }
        }
        signature = payload.get("thought_signature") or payload.get("thoughtSignature")
        if signature is not None:
            part["thought_signature"] = signature
        return part
    if block_type == "function_response":
        part = {
            "function_response": {
                "id": payload.get("id"),
                "name": payload.get("name", ""),
                "response": payload.get("response", {}) or {},
            }
        }
        signature = payload.get("thought_signature") or payload.get("thoughtSignature")
        if signature is not None:
            part["thought_signature"] = signature
        return part
    return None


def _canonical_to_google_native_message(canonical: CanonicalMessage) -> list[Any]:
    parts: list[dict[str, Any]] = []

    for block in canonical.content:
        payload = block.payload if isinstance(block.payload, dict) else None
        if block.type == "text":
            if block.text:
                parts.append({"text": block.text})
            continue
        if block.type == "reasoning":
            text = _reasoning_from_block(block)
            part = {"text": text, "thought": True}
            signature = block.signature
            if signature is None and isinstance(payload, dict):
                signature = payload.get("thought_signature") or payload.get("thoughtSignature")
            if signature is not None:
                part["thought_signature"] = signature
            parts.append(part)
            continue
        if block.type == "function_call":
            if isinstance(payload, dict):
                provider_part = _google_part_from_provider_payload(payload)
                if provider_part is not None:
                    parts.append(provider_part)
                    continue
            parts.append(
                {
                    "function_call": {
                        "id": block.call_id,
                        "name": block.name or "",
                        "args": _dict_arguments(block.arguments),
                    },
                    **({"thought_signature": block.signature} if block.signature is not None else {}),
                }
            )
            continue
        if block.type == "function_response":
            if isinstance(payload, dict):
                provider_part = _google_part_from_provider_payload(payload)
                if provider_part is not None:
                    parts.append(provider_part)
                    continue
            parts.append(
                {
                    "function_response": {
                        "id": block.call_id,
                        "name": block.name or "",
                        "response": _function_response_payload(block.output),
                    },
                    **({"thought_signature": block.signature} if block.signature is not None else {}),
                }
            )
            continue
        if block.type == "provider_item" and isinstance(payload, dict):
            provider_part = _google_part_from_provider_payload(payload)
            if provider_part is not None:
                parts.append(provider_part)

    if canonical.role == "system":
        return [{"role": "system", "parts": parts}]
    if canonical.role == "assistant":
        return [{"role": "model", "parts": parts}]
    return [{"role": "user", "parts": parts}]


def _responses_reasoning_item(block: Any) -> dict[str, Any]:
    payload = block.payload if isinstance(block.payload, dict) else None
    if isinstance(payload, dict) and payload.get("type") == "reasoning":
        return dict(payload)

    item: dict[str, Any] = {"type": "reasoning"}
    if getattr(block, "text", None):
        item["summary"] = [{"type": "summary_text", "text": getattr(block, "text")}]
    if getattr(block, "signature", None):
        item["signature"] = getattr(block, "signature")
    return item


def _canonical_to_responses_messages(canonical: CanonicalMessage) -> list[Any]:
    items: list[Any] = []
    text_fragments: list[str] = []

    for block in canonical.content:
        payload = block.payload if isinstance(block.payload, dict) else None
        if block.type == "text" and block.text:
            text_fragments.append(block.text)
            continue
        if block.type == "reasoning":
            items.append(_responses_reasoning_item(block))
            continue
        if block.type == "function_call":
            if isinstance(payload, dict) and payload.get("type") == "function_call":
                items.append(dict(payload))
            else:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.call_id,
                        "name": block.name or "",
                        "arguments": _json_arguments(block.arguments),
                    }
                )
            continue
        if block.type == "function_response":
            if isinstance(payload, dict) and payload.get("type") == "function_call_output":
                items.append(dict(payload))
            else:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.call_id,
                        "output": _stringify(block.output),
                    }
                )
            continue
        if block.type == "provider_item" and isinstance(payload, dict):
            items.append(dict(payload))

    if text_fragments:
        text = "".join(text_fragments)
        if canonical.role == "assistant":
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        else:
            items.append({"role": canonical.role, "content": text})

    if not items:
        items.append({"role": canonical.role, "content": ""})
    return items


def _canonical_to_replay_entries(canonical: CanonicalMessage, provider_name: Optional[str]) -> list[Any]:
    provider = _normalize_provider_name(provider_name)

    if provider == "openai_responses":
        return _canonical_to_responses_messages(canonical)
    if provider == "google_native":
        return _canonical_to_google_native_message(canonical)
    if provider == "anthropic_native":
        return _canonical_to_anthropic_message(canonical, include_reasoning_sidecar=False)
    if provider == "anthropic":
        return _canonical_to_anthropic_message(canonical, include_reasoning_sidecar=True)
    if provider == "google":
        return _canonical_to_openai_like_message(canonical, tool_role="function", preserve_reasoning=True)
    return _canonical_to_openai_like_message(canonical, tool_role="tool", preserve_reasoning=True)


def _is_google_function_response_turn(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") != "user":
        return False
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        return False
    return all(isinstance(part, dict) and "function_response" in part for part in parts)


def _is_anthropic_tool_result_turn(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def append_replay_entry(prepared: list[Any], entry: Any, provider_name: Optional[str]) -> None:
    provider = _normalize_provider_name(provider_name)
    if provider == "google_native":
        if _is_google_function_response_turn(entry) and prepared and _is_google_function_response_turn(prepared[-1]):
            prepared[-1]["parts"].extend(entry["parts"])
            return
    if provider in {"anthropic", "anthropic_native"}:
        if _is_anthropic_tool_result_turn(entry) and prepared and _is_anthropic_tool_result_turn(prepared[-1]):
            prepared[-1]["content"].extend(entry["content"])
            return
    prepared.append(entry)


def canonical_to_replay_history(messages: list[Any], provider_name: Optional[str]) -> list[Any]:
    prepared: list[Any] = []
    for message in messages:
        canonical = coerce_canonical_message(message)
        canonical_entries = [canonical] if canonical is not None else _history_entry_to_canonical(
            message,
            provider_name,
        )
        for canonical_entry in canonical_entries:
            for entry in _canonical_to_replay_entries(canonical_entry, provider_name):
                append_replay_entry(prepared, entry, provider_name)
    return prepared


def _history_entry_to_canonical(message: Any, provider_name: Optional[str]) -> list[CanonicalMessage]:
    from .providers import create_codec

    return create_codec(provider_name).history_entry_to_canonical(message)
