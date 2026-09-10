"""Public extension boundary for request assembly and history compaction."""

from abc import ABC, abstractmethod
from core.providers import create_codec
from .token.budget import ContextBudgetExceeded


class BaseContextManager(ABC):
    @property
    @abstractmethod
    def budget(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def counter(self):
        raise NotImplementedError

    @abstractmethod
    def build_request_input(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compact_persistent_history(self, *args, **kwargs):
        raise NotImplementedError

    async def acompact_persistent_history(self, *args, **kwargs):
        import asyncio
        return await asyncio.to_thread(self.compact_persistent_history, *args, **kwargs)

    def count_request(self, request, *, tools=None, reasoning=None):
        return create_codec(request.provider_name).count_request_tokens(
            self.counter, request.replay_history, system_prompt=request.render_system_prompt(),
            tools=tools, reasoning=reasoning,
        )

    def estimate_request(self, request, *, usage_tracker=None, llm=None, history_epoch=0, tools=None, reasoning=None, options=None):
        if usage_tracker is not None:
            return usage_tracker.estimate(self, request, llm=llm, history_epoch=history_epoch,
                                          tools=tools, reasoning=reasoning, options=options)
        from .usage import ContextTokenEstimate
        tokens = self.count_request(request, tools=tools, reasoning=reasoning)
        return ContextTokenEstimate(tokens, tokens, "local_request_estimate", {})

    def validate_request(self, request, *, tools=None, reasoning=None, usage_tracker=None, llm=None, history_epoch=0, options=None):
        tokens = self.estimate_request(request, usage_tracker=usage_tracker, llm=llm, history_epoch=history_epoch,
                                       tools=tools, reasoning=reasoning, options=options).tokens
        limit = self.budget.max_tokens
        if limit is not None and tokens > limit:
            raise ContextBudgetExceeded(tokens, limit)
        return tokens
