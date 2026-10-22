import pytest

from easyagent.context import ContextManager, LLMHistoryCompactor, RuleBasedHistoryCompactor
from core.providers import create_codec


class Summary:
    def __init__(self):
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages)
        return '["Task and completed work summary"]'

    async def ainvoke(self, messages):
        return self.invoke(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
async def test_zero_recent_compacts_single_turn_and_keeps_temporary_injections(asynchronous):
    manager = ContextManager(max_tokens=1000)
    summarizer = Summary()
    manager.set_history_compactor(LLMHistoryCompactor(llm=summarizer, recent_turns=0))
    codec = create_codec('openai')
    canonical = codec.replay_to_canonical([
        {'role': 'user', 'content': 'user task'},
        {'role': 'assistant', 'content': 'finished work ' * 500},
    ])
    temporary = codec.replay_to_canonical([{'role': 'user', 'content': 'active skill instruction'}])[0]
    temporary.metadata.update(is_meta=True, metaMessageLifecycle='invocation')
    canonical.append(temporary)
    args = (canonical, codec.canonical_to_replay(canonical))
    kwargs = dict(provider_name='openai', token_counter=manager.counter, max_tokens=1000, force=True)
    result = (await manager.acompact_persistent_history(*args, **kwargs) if asynchronous
              else manager.compact_persistent_history(*args, **kwargs))
    assert manager.history_compactor.recent_turns == 0
    assert result.compaction_possible and result.was_compacted
    assert len(summarizer.requests) == 1
    assert 'user task' in str(summarizer.requests[0])
    assert 'finished work' in str(summarizer.requests[0])
    assert 'active skill instruction' not in str(summarizer.requests[0])
    assert 'active skill instruction' in str(result.replay_history)
    assert 'Task and completed work summary' in str(result.replay_history)
    assert 'finished work' not in str(result.replay_history)


def test_positive_defaults_and_zero_rule_compaction():
    assert LLMHistoryCompactor(llm=Summary()).recent_turns == 4
    assert RuleBasedHistoryCompactor().recent_turns == 4
    manager = ContextManager(max_tokens=1000)
    manager.set_history_compactor(RuleBasedHistoryCompactor(recent_turns=0))
    codec = create_codec('openai')
    replay = [
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call1', 'type': 'function', 'function': {'name': 'Read', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call1', 'name': 'Read', 'content': 'large output ' * 1000},
    ]
    result = manager.compact_persistent_history(codec.replay_to_canonical(replay), replay,
        provider_name='openai', token_counter=manager.counter, max_tokens=1000, force=True)
    assert result.compaction_possible and result.was_compacted
    assert len(result.replay_history[-1]['content']) < len(replay[-1]['content'])
