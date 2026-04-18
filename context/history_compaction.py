from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.history import CanonicalMessage, _json_safe, coerce_canonical_message
from core.providers import create_codec
from core.request_input import ReplayRequestInput


@dataclass
class HistoryCompactionResult:
    canonical_history: list[Any]
    replay_history: list[Any]
    was_compacted: bool
    compaction_possible: bool
    tokens_before: int
    tokens_after: int
    budget: int


def _copy_entry(value: Any) -> Any:
    if isinstance(value, dict):
        return _json_safe(value)
    if hasattr(value, "to_dict"):
        try:
            payload = value.to_dict()
            if isinstance(payload, dict):
                return _json_safe(payload)
        except Exception:
            pass
    return value


def _to_canonical_history(entries: list[Any], provider_name: Optional[str]) -> list[Any]:
    codec = create_codec(provider_name)
    return codec.replay_to_canonical(entries or [])


def _group_turns(history: list[Any]) -> list[list[Any]]:
    turns: list[list[Any]] = []
    current: list[Any] = []
    for message in history:
        canonical = coerce_canonical_message(message)
        role = canonical.role if canonical is not None else None
        if role == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def _split_turn_tail(history: list[Any], preserve_tail_turns: int) -> tuple[list[Any], list[Any]]:
    if not history:
        return [], []
    turns = _group_turns(history)
    if preserve_tail_turns <= 0 or preserve_tail_turns >= len(turns):
        return [], [_copy_entry(message) for message in history]
    prefix_turns = turns[:-preserve_tail_turns]
    tail_turns = turns[-preserve_tail_turns:]
    prefix = [_copy_entry(message) for turn in prefix_turns for message in turn]
    tail = [_copy_entry(message) for turn in tail_turns for message in turn]
    return prefix, tail


def _count_request_tokens(
    provider_name: Optional[str],
    counter: Any,
    replay_history: list[Any],
    *,
    system_prompt: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning: Optional[dict[str, Any]] = None,
    pending_messages: Optional[list[Any]] = None,
) -> int:
    codec = create_codec(provider_name)
    return codec.count_request_tokens(
        counter,
        replay_history,
        system_prompt=system_prompt,
        tools=tools,
        pending_messages=pending_messages,
        reasoning=reasoning,
    )


def _compactor_recent_turns(history_compactor: Any) -> int:
    return max(1, int(getattr(history_compactor, "recent_turns", 1) or 1))


def _result(
    *,
    canonical_history: list[Any],
    replay_history: list[Any],
    was_compacted: bool,
    compaction_possible: bool,
    tokens_before: int,
    tokens_after: int,
    budget: int,
) -> HistoryCompactionResult:
    return HistoryCompactionResult(
        canonical_history=canonical_history,
        replay_history=replay_history,
        was_compacted=was_compacted,
        compaction_possible=compaction_possible,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=budget,
    )


def compact_request_input(
    request_input: ReplayRequestInput,
    *,
    token_counter: Any,
    history_compactor: Any,
    max_tokens: int,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning: Optional[dict[str, Any]] = None,
) -> HistoryCompactionResult:
    provider_name = request_input.provider_name
    replay_history = list(request_input.replay_history)
    tokens_before = _count_request_tokens(
        provider_name,
        token_counter,
        replay_history,
        system_prompt=request_input.system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    if tokens_before <= max_tokens:
        return _result(
            canonical_history=_to_canonical_history(replay_history, provider_name),
            replay_history=replay_history,
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    canonical_history = _to_canonical_history(replay_history, provider_name)
    preserve_tail_turns = _compactor_recent_turns(history_compactor)
    canonical_prefix, canonical_tail = _split_turn_tail(canonical_history, preserve_tail_turns)
    if not canonical_prefix:
        return _result(
            canonical_history=canonical_history,
            replay_history=replay_history,
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    codec = create_codec(provider_name)
    replay_tail = codec.canonical_to_replay(canonical_tail)
    tail_tokens = _count_request_tokens(
        provider_name,
        token_counter,
        replay_tail,
        system_prompt=request_input.system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    prefix_budget = max(0, max_tokens - tail_tokens)
    compacted_prefix = history_compactor.compact(canonical_prefix, max_tokens=prefix_budget)
    compacted_canonical = [*compacted_prefix, *canonical_tail]
    compacted_replay = codec.canonical_to_replay(compacted_canonical)
    tokens_after = _count_request_tokens(
        provider_name,
        token_counter,
        compacted_replay,
        system_prompt=request_input.system_prompt,
        tools=tools,
        reasoning=reasoning,
    )

    if tokens_after > max_tokens and compacted_prefix:
        overflow = tokens_after - max_tokens
        reduced_budget = max(0, prefix_budget - overflow)
        if reduced_budget < prefix_budget:
            compacted_prefix = history_compactor.compact(canonical_prefix, max_tokens=reduced_budget)
            compacted_canonical = [*compacted_prefix, *canonical_tail]
            compacted_replay = codec.canonical_to_replay(compacted_canonical)
            tokens_after = _count_request_tokens(
                provider_name,
                token_counter,
                compacted_replay,
                system_prompt=request_input.system_prompt,
                tools=tools,
                reasoning=reasoning,
            )

    request_input.set_replay_history(compacted_replay)
    return _result(
        canonical_history=compacted_canonical,
        replay_history=compacted_replay,
        was_compacted=compacted_replay != replay_history,
        compaction_possible=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=max_tokens,
    )


async def acompact_request_input(
    request_input: ReplayRequestInput,
    *,
    token_counter: Any,
    history_compactor: Any,
    max_tokens: int,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning: Optional[dict[str, Any]] = None,
) -> HistoryCompactionResult:
    provider_name = request_input.provider_name
    replay_history = list(request_input.replay_history)
    tokens_before = _count_request_tokens(
        provider_name,
        token_counter,
        replay_history,
        system_prompt=request_input.system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    if tokens_before <= max_tokens:
        return _result(
            canonical_history=_to_canonical_history(replay_history, provider_name),
            replay_history=replay_history,
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    canonical_history = _to_canonical_history(replay_history, provider_name)
    preserve_tail_turns = _compactor_recent_turns(history_compactor)
    canonical_prefix, canonical_tail = _split_turn_tail(canonical_history, preserve_tail_turns)
    if not canonical_prefix:
        return _result(
            canonical_history=canonical_history,
            replay_history=replay_history,
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    codec = create_codec(provider_name)
    replay_tail = codec.canonical_to_replay(canonical_tail)
    tail_tokens = _count_request_tokens(
        provider_name,
        token_counter,
        replay_tail,
        system_prompt=request_input.system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    prefix_budget = max(0, max_tokens - tail_tokens)
    compacted_prefix = await history_compactor.acompact(canonical_prefix, max_tokens=prefix_budget)
    compacted_canonical = [*compacted_prefix, *canonical_tail]
    compacted_replay = codec.canonical_to_replay(compacted_canonical)
    tokens_after = _count_request_tokens(
        provider_name,
        token_counter,
        compacted_replay,
        system_prompt=request_input.system_prompt,
        tools=tools,
        reasoning=reasoning,
    )

    if tokens_after > max_tokens and compacted_prefix:
        overflow = tokens_after - max_tokens
        reduced_budget = max(0, prefix_budget - overflow)
        if reduced_budget < prefix_budget:
            compacted_prefix = await history_compactor.acompact(canonical_prefix, max_tokens=reduced_budget)
            compacted_canonical = [*compacted_prefix, *canonical_tail]
            compacted_replay = codec.canonical_to_replay(compacted_canonical)
            tokens_after = _count_request_tokens(
                provider_name,
                token_counter,
                compacted_replay,
                system_prompt=request_input.system_prompt,
                tools=tools,
                reasoning=reasoning,
            )

    request_input.set_replay_history(compacted_replay)
    return _result(
        canonical_history=compacted_canonical,
        replay_history=compacted_replay,
        was_compacted=compacted_replay != replay_history,
        compaction_possible=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=max_tokens,
    )


def compact_persistent_history(
    canonical_history: list[Any],
    replay_history: list[Any],
    *,
    provider_name: Optional[str],
    token_counter: Any,
    history_compactor: Any,
    system_prompt: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning: Optional[dict[str, Any]] = None,
    max_tokens: int,
) -> HistoryCompactionResult:
    tokens_before = _count_request_tokens(
        provider_name,
        token_counter,
        replay_history,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    if tokens_before <= max_tokens:
        return _result(
            canonical_history=[_copy_entry(item) for item in canonical_history],
            replay_history=[_copy_entry(item) for item in replay_history],
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    preserve_tail_turns = _compactor_recent_turns(history_compactor)
    canonical_prefix, canonical_tail = _split_turn_tail(canonical_history, preserve_tail_turns)
    if not canonical_prefix:
        return _result(
            canonical_history=[_copy_entry(item) for item in canonical_history],
            replay_history=[_copy_entry(item) for item in replay_history],
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    codec = create_codec(provider_name)
    replay_tail = codec.canonical_to_replay(canonical_tail)
    tail_tokens = _count_request_tokens(
        provider_name,
        token_counter,
        replay_tail,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    prefix_budget = max(0, max_tokens - tail_tokens)
    compacted_prefix = history_compactor.compact(canonical_prefix, max_tokens=prefix_budget)
    compacted_canonical = [*compacted_prefix, *canonical_tail]
    compacted_replay = codec.canonical_to_replay(compacted_canonical)
    tokens_after = _count_request_tokens(
        provider_name,
        token_counter,
        compacted_replay,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
    )

    if tokens_after > max_tokens and compacted_prefix:
        overflow = tokens_after - max_tokens
        reduced_budget = max(0, prefix_budget - overflow)
        if reduced_budget < prefix_budget:
            compacted_prefix = history_compactor.compact(canonical_prefix, max_tokens=reduced_budget)
            compacted_canonical = [*compacted_prefix, *canonical_tail]
            compacted_replay = codec.canonical_to_replay(compacted_canonical)
            tokens_after = _count_request_tokens(
                provider_name,
                token_counter,
                compacted_replay,
                system_prompt=system_prompt,
                tools=tools,
                reasoning=reasoning,
            )

    return _result(
        canonical_history=compacted_canonical,
        replay_history=compacted_replay,
        was_compacted=compacted_canonical != canonical_history,
        compaction_possible=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=max_tokens,
    )


async def acompact_persistent_history(
    canonical_history: list[Any],
    replay_history: list[Any],
    *,
    provider_name: Optional[str],
    token_counter: Any,
    history_compactor: Any,
    system_prompt: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    reasoning: Optional[dict[str, Any]] = None,
    max_tokens: int,
) -> HistoryCompactionResult:
    tokens_before = _count_request_tokens(
        provider_name,
        token_counter,
        replay_history,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    if tokens_before <= max_tokens:
        return _result(
            canonical_history=[_copy_entry(item) for item in canonical_history],
            replay_history=[_copy_entry(item) for item in replay_history],
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    preserve_tail_turns = _compactor_recent_turns(history_compactor)
    canonical_prefix, canonical_tail = _split_turn_tail(canonical_history, preserve_tail_turns)
    if not canonical_prefix:
        return _result(
            canonical_history=[_copy_entry(item) for item in canonical_history],
            replay_history=[_copy_entry(item) for item in replay_history],
            was_compacted=False,
            compaction_possible=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=max_tokens,
        )

    codec = create_codec(provider_name)
    replay_tail = codec.canonical_to_replay(canonical_tail)
    tail_tokens = _count_request_tokens(
        provider_name,
        token_counter,
        replay_tail,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
    )
    prefix_budget = max(0, max_tokens - tail_tokens)
    compacted_prefix = await history_compactor.acompact(canonical_prefix, max_tokens=prefix_budget)
    compacted_canonical = [*compacted_prefix, *canonical_tail]
    compacted_replay = codec.canonical_to_replay(compacted_canonical)
    tokens_after = _count_request_tokens(
        provider_name,
        token_counter,
        compacted_replay,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
    )

    if tokens_after > max_tokens and compacted_prefix:
        overflow = tokens_after - max_tokens
        reduced_budget = max(0, prefix_budget - overflow)
        if reduced_budget < prefix_budget:
            compacted_prefix = await history_compactor.acompact(canonical_prefix, max_tokens=reduced_budget)
            compacted_canonical = [*compacted_prefix, *canonical_tail]
            compacted_replay = codec.canonical_to_replay(compacted_canonical)
            tokens_after = _count_request_tokens(
                provider_name,
                token_counter,
                compacted_replay,
                system_prompt=system_prompt,
                tools=tools,
                reasoning=reasoning,
            )

    return _result(
        canonical_history=compacted_canonical,
        replay_history=compacted_replay,
        was_compacted=compacted_canonical != canonical_history,
        compaction_possible=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=max_tokens,
    )
