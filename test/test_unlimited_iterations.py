import pytest
from executor import DefaultAgentExecutor
from test_streaming_tool_calls import (ScriptedStreamingProvider, TestAgentStreamingContract as StreamingContract,
    _openai_text_chunk, _openai_tool_call_chunk)


class ManyRounds(ScriptedStreamingProvider):
    def invoke_raw(self, request):
        return self._round_chunks()[-1].choices[0].delta

    async def async_invoke_raw(self, request):
        return self.invoke_raw(request)

    def _round_chunks(self):
        self.round += 1
        if self.round <= 16:
            return [_openai_tool_call_chunk(f'call_{self.round}', 'echo', '{"text":"step"}')]
        return [_openai_text_chunk('finished', finish_reason='stop')]


def test_unlimited_sync_stream_exceeds_previous_limits():
    agent = StreamingContract._tool_agent(ManyRounds())
    events = list(agent.stream('work', max_iter=None))
    assert events[-1].content == 'finished'
    assert sum(e.type.value == 'tool_result' for e in events) == 16


def test_unlimited_invoke():
    agent = StreamingContract._tool_agent(ManyRounds())
    assert agent.invoke('work', max_iter=None) == 'finished'


@pytest.mark.asyncio
async def test_unlimited_ainvoke():
    agent = StreamingContract._tool_agent(ManyRounds())
    assert await agent.ainvoke('work', max_iter=None) == 'finished'


@pytest.mark.asyncio
async def test_unlimited_async_stream_and_finite_limit():
    agent = StreamingContract._tool_agent(ManyRounds())
    events = [event async for event in agent.astream('work', max_iter=None)]
    assert events[-1].content == 'finished'
    assert sum(e.type.value == 'tool_result' for e in events) == 16
    agent = StreamingContract._tool_agent(ManyRounds())
    with pytest.raises(RuntimeError, match='max_iter=2'):
        async for _ in agent.astream('work', max_iter=2):
            pass
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            DefaultAgentExecutor._validate('work', invalid, .2)
