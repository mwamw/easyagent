# Agent Runtime

Every Agent owns one `AgentRuntime`. It contains workspace, session identity,
execution mode, current task, cancellation state, and the fixed event, Hook,
MetaMessage and interruption facilities. It is created with the Agent and remains
the same object until the Agent closes. It is not a `with_*` functional module.

```python
from easyagent import BasicAgent, AgentRuntime

agent = BasicAgent("assistant", llm)
print(agent.runtime.workspace_root)
agent.runtime.event_bus.subscribe(handle_event)

child_runtime = agent.runtime.derive(
    workspace_root="/workspace/worker",
    allowed_roots=("/workspace/worker",),
    agent_id="worker",
)
child = BasicAgent("worker", llm, runtime=child_runtime)
```

Derivation creates fresh locks, session identity, events, hooks, messages and
interruptions. Metadata is deep-copied. It does not copy active invocations,
cancellation requests, or pending approvals. Sharing an owned Runtime with a
second Agent raises an error. Agent.runtime and its fixed facilities cannot be
replaced after construction.

## Module dependencies

Agent composition injects `runtime` and any required module references. Skill and
Multi-Agent receive `runtime` and `tool_registry`. Plan receives the same pair,
with an optional registry. Modules can read `runtime.workspace_root`, publish
through `runtime.event_bus`, and inject through `runtime.metamessages`.

Plan changes `runtime.execution_mode` and `runtime.plan_allowed_actions`.
Authorization receives these values in a typed `AuthorizationContext`; the
installed permission engine decides what they mean. Rules and approval presets
remain owned by the permission implementation.

Context still owns history and input assembly. Memory owns memory contents.
Runtime does not act as a registry of functional modules.

## Invocation and lifecycle

`AgentExecutionServices` supplies one `runtime` alongside Executor's module
dependencies. A custom Executor can call `runtime.begin_invocation(query,
stream=...)` and must call `runtime.end_invocation()` in its cleanup path.
`AgentInvocationState` holds the per-call phase, round, output and error;
`runtime.active_invocation` and `last_invocation` expose the relevant instance.

The default Executor supports one active invocation per Agent. Session operations
reserve the same runtime lock, including across asynchronous storage I/O.
Cancellation and failed I/O release the reservation. Closing an active Agent is
rejected. After its modules close, the Runtime releases subscriptions and message
bindings.

## Session and migration

`SessionManager` receives the actual Runtime plus `SessionBindings` for Context
and tool queries. Restoring a session keeps live facilities and module bindings;
it does not instantiate an Agent from serialized class names. Workspace selection
remains an explicit assembly decision.

The old `ExecutionContext` type and `agent.execution_context` are removed.
Use `AgentRuntime` and `agent.runtime`. Prompt contributors use `context.runtime`.
Custom Skill/Multi-Agent bindings no longer receive separate event and message
managers. `SessionRuntime` is replaced by `SessionBindings`. Hooks are registered
into the Runtime's existing pipeline; supplying a HookManager to `with_hooks`
registers its hook objects without replacing that pipeline.
