# Core module boundaries

This refactor changes EasyAgent composition, not the LLM implementation. `EasyLLM`, its providers,
response formats and `invoke`/`ainvoke`/streaming interfaces remain unchanged.

## Required and optional capabilities

Every `BasicAgent` has a supplied LLM, a default Executor, a System Prompt composer and an owned Context.
The default Context has no total token limit. Tools, permissions, callbacks, memory, skills,
planning, observability and external services are optional. Install permissions and guardrails explicitly
when enabling tools that can affect the environment:

```python
agent.with_permissions().with_hooks()
```

No permission module means no Agent-level authorization check; it is not a sandbox or a hidden policy.
The Runtime always owns an initially empty Hook pipeline; `with_hooks` registers hooks in that pipeline.

## Ownership and public calls

- Agent installs modules and binds collaborators. It no longer owns a second history object or token tracker.
- Executor directly coordinates the prompt composer and the explicit Context stages
  (`process_meta_messages`, `build_input`, `calculate_usage`, compaction,
  validation, and usage observation),
  existing LLM methods, and the public tool/permission interfaces.
- Context owns framework `ConversationHistory` and complete input preparation. Provider replay,
  token usage and compaction are built-in implementation details under `context/default/`;
  a custom Context need not expose `builder`, `counter` or `budget`.
- Modules contribute instructions through `get_prompt_blocks(context)`. Agent registers these providers
  by owner name; replacement removes stale contributions. Composer replacement does not lose providers.
- MetaMessage injects/removes messages through the Context contract. Session persistence exports and
  restores Context state without constructing the default history implementation.
- Agent tool execution occurs only inside the installed executor's ReAct loop through
  `execute_tool_call()` / `aexecute_tool_call()`. BaseAgent exposes no parallel direct or approved
  execution path. A host resolving an interruption supplies the externally obtained observation,
  then starts the next invocation.

## Context lifecycle

A Context is stateful and cannot be shared between Agent instances. Create one per Agent, even when
sharing an LLM. Replacing Context on an idle Agent transfers canonical history through public methods;
replacement during an active invoke is rejected. A custom implementation can reject history it cannot
represent. Failed transfer leaves the Agent bound to the original Context.

Session snapshots retain the existing top-level `history` field for serialized Context state. This is
the only copy; it is no longer separately duplicated in `modules.context`. Custom Context snapshots
may have a different internal structure and validate it through `validate_state()`.

## Import and API migration

| Previous internal path or assumption | Current interface |
| --- | --- |
| `agent.components.executor` | `easyagent.executors` (`executor/base.py`, `executor/default.py`) |
| `agent.components.prompt_composer` | `easyagent.prompting` (`prompt/base.py`, `prompt/composer.py`) |
| `agent.components.conversation_history` | Framework `ConversationHistory` at `context/history.py` |
| `agent.components.tool_interrupt_controller` | `runtime.interruptions` |
| External `RequestPreparer` | Explicit `BaseContextManager` stage contracts coordinated by Executor |
| `agent.history_store` | Agent history accessors, or public operations on `agent.context_manager` |
| `AgentExecutionServices.history` | `AgentExecutionServices.context_manager` |
| `context_manager is None` on a minimal Agent | Context always exists |
| Implicit permissions / guardrails | `with_permissions()` / `with_hooks()` |

`PromptBuildContext` exposes request facts and registered providers, not concrete Memory/Skill/Plan
objects. `include_defaults` controls default behavior text; `include_module_blocks` independently
controls module contributions. AgentRuntime now owns shared state and fixed runtime facilities. Functional modules remain separate.

## Regression gates

`test/test_module_ownership.py` runs a non-default Context with framework history through all four invocation APIs, message
injection and session restore. It also verifies optional modules, prompt replacement, public custom
executor dispatch, resource cleanup and prohibited cross-module imports. Existing provider-usage,
compaction, cancellation, tool-interruption and session tests continue to protect behavior.
