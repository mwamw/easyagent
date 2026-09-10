"""One request preparation algorithm, driven synchronously or asynchronously."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from context.manager import ContextManager
from context.builder import ContextBuilder
from core.Exception import HookExecutionError, AgentStopRequested
from core.request_compiler import compile_prompt_blocks
from core.scope import RuntimeScope


@dataclass
class _Operation:
    sync: Callable
    async_: Callable


class RequestPreparer:
    """Coordinates hooks/assembly; content producers inject their own messages."""

    @staticmethod
    def context_manager(services):
        if services.context_manager is not None:
            return services.context_manager
        tracker = services.context_usage
        return ContextManager(builder=ContextBuilder(counter=tracker.counter)) if tracker is not None else ContextManager()

    def build(self, services, state, *, temperature, kwargs, on_compaction, tool_payload):
        steps = self._steps(services, state, temperature, kwargs, on_compaction, tool_payload)
        result = None
        try:
            while True:
                try:
                    operation = steps.send(result)
                except StopIteration as done:
                    return done.value
                result = operation.sync()
        finally:
            steps.close()

    async def abuild(self, services, state, *, temperature, kwargs, on_compaction, tool_payload):
        steps = self._steps(services, state, temperature, kwargs, on_compaction, tool_payload)
        result = None
        try:
            while True:
                try:
                    operation = steps.send(result)
                except StopIteration as done:
                    return done.value
                result = await operation.async_()
        finally:
            steps.close()

    @staticmethod
    def _hook(services, stage, payload):
        return _Operation(
            lambda: getattr(services.hook_manager, stage)(payload),
            lambda: services.hook_manager.arun_stage(stage, payload),
        )

    @staticmethod
    def _check_hook(hook, stage):
        if hook.blocked:
            raise HookExecutionError(hook.message or f"Request blocked at {stage}",
                                     stage=stage, error_type=hook.error_type,
                                     metadata={"audit": hook.audit})

    def _steps(self, services, state, temperature, kwargs, on_compaction, tool_payload):
        services.mailbox_sync()
        preparation = yield self._hook(services, "before_context", {
            "query": state.query,
            "history": services.history.canonical,
            "scope": RuntimeScope(services.session_id, services.execution_context.workspace_root),
            "invocation_id": state.invocation_id,
            "request_id": f"{state.invocation_id}:{state.round_number}",
            "round_number": state.round_number,
        })
        self._check_hook(preparation, "before_context")
        services.metamessage_manager.flush()
        blocks = services.prompt_composer.compose(services.prompt_context_factory(state.query))
        compiled = compile_prompt_blocks(blocks, cache_policy=services.config.cache_policy)
        # Provider tool formatting remains the executor's responsibility.
        tools = tool_payload()
        manager = self.context_manager(services)
        request_options = {"temperature": temperature, "reasoning": services.reasoning, **kwargs}
        epoch = services.history.context_epoch
        estimate_arguments = dict(usage_tracker=services.context_usage, llm=services.llm,
                                  history_epoch=epoch, tools=tools, reasoning=services.reasoning,
                                  options=request_options)
        arguments = dict(
            query=state.query, include_query=False,
            provider_name=services.llm.provider_name,
            system_prompt=compiled.system_prompt,
            system_prompt_blocks=compiled.system_prompt_blocks,
            system_reminder_blocks=compiled.system_reminder_blocks,
            dynamic_tail_blocks=compiled.dynamic_tail_blocks,
            on_demand_expansion_blocks=compiled.on_demand_expansion_blocks,
            cache_policy=compiled.cache_policy, tools=tools, reasoning=services.reasoning,
        )
        request = manager.build_request_input(replay_history=services.history.replay, **arguments)
        estimate = manager.estimate_request(request, **estimate_arguments)
        total = estimate.tokens
        limit = manager.budget.max_tokens
        compaction = None
        if limit is not None and total > limit:
            hook = yield self._hook(services, "before_compaction", {
                "canonical_history": services.history.canonical,
                "replay_history": services.history.replay,
                "max_tokens": limit, "estimated_tokens": total,
                "token_estimate_source": estimate.source,
                "token_estimate": estimate.metadata,
                "invocation_id": state.invocation_id,
            })
            self._check_hook(hook, "before_compaction")
            base_tokens = services.llm.count_request_tokens(
                manager.counter, services.history.replay, system_prompt=compiled.system_prompt,
                tools=tools, reasoning=services.reasoning,
            )
            inputs = (hook.payload.get("canonical_history", services.history.canonical),
                      hook.payload.get("replay_history", services.history.replay))
            overhead = max(0, estimate.local_tokens - base_tokens)
            options = dict(provider_name=services.llm.provider_name, token_counter=manager.counter,
                           system_prompt=compiled.system_prompt, tools=tools, reasoning=services.reasoning,
                           max_tokens=max(0, min(limit, int(hook.payload.get("max_tokens", limit))) - overhead),
                           tokens_before_override=max(0, total - overhead),
                           metadata={"token_estimate": {"source": estimate.source, **estimate.metadata}})
            compaction = yield _Operation(
                lambda: manager.compact_persistent_history(*inputs, **options),
                lambda: manager.acompact_persistent_history(*inputs, **options),
            )
            if compaction.was_compacted:
                request = manager.build_request_input(replay_history=compaction.replay_history, **arguments)
                # The old API anchor cannot be applied to a newly compacted prefix.
                epoch += 1
        options = {"temperature": temperature, "reasoning": services.reasoning, **kwargs}
        hook = yield self._hook(services, "before_llm_request", {
            "request_input": request, "tools": tools, "options": dict(options),
            "invocation_id": state.invocation_id, "round_number": state.round_number,
        })
        self._check_hook(hook, "before_llm_request")
        request = hook.payload.get("request_input", request)
        tools = hook.payload.get("tools", tools)
        options = dict(hook.payload.get("options", options))
        manager.validate_request(request, tools=tools, reasoning=options.get("reasoning"),
                                 usage_tracker=services.context_usage, llm=services.llm,
                                 history_epoch=epoch, options=options)
        reason = services.stop_checker()
        if reason is not None:
            raise AgentStopRequested(reason)
        if compaction is not None and compaction.was_compacted:
            services.history.replace(compaction.canonical_history)
            on_compaction(compaction)
        if services.context_usage is not None:
            state.usage_request = services.context_usage.prepare(
                manager, request, llm=services.llm, history_epoch=services.history.context_epoch,
                tools=tools, reasoning=options.get("reasoning"), options=options, query=state.query,
            )
        return request, tools, options
