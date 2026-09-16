"""Default Context token estimation helpers; not part of the module contract."""
from core.providers import create_codec
from .token.budget import ContextBudgetExceeded

class RequestEstimation:
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
