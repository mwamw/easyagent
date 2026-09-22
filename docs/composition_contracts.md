# Module composition contracts

EasyAgent distinguishes fixed mechanisms from replaceable policies. Skill, MCP,
and ToolRegistry remain fixed framework mechanisms, extended through skills,
servers, and tools. They do not require alternative manager base classes.

## Explicit dependencies

`with_*` installs modules and injects their dependencies. A module never receives
the whole Agent to discover private fields. CodeIntel receives `AgentRuntime`;
Plan receives Runtime and, when available, ToolRegistry. The optional module
contracts are available through the public exports:

```python
from easyagent.codeintel import BaseCodeIntel
from easyagent.tasks import BaseTaskService
from easyagent.worktree import BaseWorktreeManager
from easyagent.plans import BasePlanMode
```

`BaseCodeIntel` describes code queries, not a particular LSP or cache. A task
implementation supplies create/get/update/list operations without inheriting the
built-in storage-backed service. Worktree lifecycle is defined in `worktree/base.py`,
with Git execution in `worktree/default.py` and shared values in `worktree/models.py`.

## Planning and permissions

Plan supplies `get_tools()` after binding. Agent registers those tools without
reading implementation configuration or calling an undeclared installation method.
`request_exit()` requests review; it does not silently authorize execution.
The default Plan creates an independent child through `create_child()`.

Permission engines receive `PermissionRequest` and return `PermissionDecision`.
Optional `bind(runtime)` and `set_mode(mode)` are explicit contracts. An engine
without mode support rejects mode switching with `NotImplementedError`.
Rule-based matching and rule management stay on the concrete permission engine;
`BaseAgent.add_permission_rule()` has been removed. Call the installed rule
engine's `add_rule(...)` when that is the policy implementation you chose.

The default permission engine uses Runtime as its bound mode authority. Changing
the mode does not rely on a later prompt build to synchronize a second state copy.

## Child agents and resource ownership

Child assembly belongs to the multi-agent harness. Neither `Tool` nor
`BasePermissionEngine` requires or supplies `create_child()`. A single-agent
application has no child construction responsibilities.

### Tools: reuse capabilities, bind state explicitly

The default child factory creates an independent ToolRegistry. Stateless tools,
ordinary function tools, and MCP capabilities may reuse the same tool instance.
Registration visibility, deferred exposure/expansion, and permission overrides
belong to each registry; registering a shared tool does not mutate its definition.

Workspace tools (including List and NotebookEdit) are reconstructed for the child.
Bash and its output/stop tools use a child-local process manager. Parent-bound
Skill, Plan, Task, deferred-schema, Config, CodeIntel, Memory, Worktree, and
collaboration tools are not blindly inherited. Plan, Skill, Task and deferred-schema
tools are contributed again when their child components are installed. Other
module capabilities need explicit child assembly. MCP connections are borrowed:
closing a child does not close its parent's connection.

Custom tools need only implement normal Tool behavior. If a tool holds mutable
per-agent state or captures parent dependencies, provide a binding at multi-agent
assembly, not a method on the Tool class:

```python
agent.with_multi_agent(
    child_tool_factories={
        "ProjectTool": lambda tool, runtime: ProjectTool(
            workspace_root=runtime.workspace_root,
        ),
        "ParentOnlyTool": lambda tool, runtime: None,
    },
)
```

A binding returns a same-name Tool, or `None` to omit it. Reuse is permitted when
intentional; arbitrary objects are never deep-copied. Unknown workspace-tool
subclasses and compatibility delegators require explicit bindings rather than
silently losing custom behavior or retaining a parent workspace. Tool allow/deny
lists are applied after child module assembly; availability never replaces
per-call authorization.

Reuse does **not** make arbitrary custom tool state thread-safe. Shared tools
must be safe for concurrent calls, or use a factory to supply separate instances.
This change retains `run(parameters)` / `arun(parameters)`; it does not add a
mandatory ToolUseContext argument or promise isolation of all global backend state.

### Permissions: no child lifecycle on the base contract

The default multi-agent permission factory snapshots the built-in engine's public
policy data, removes parent grants, and uses the child's mode/workspace. It neither
copies connection/lock objects nor rebinds the parent's engine. This is a policy
snapshot at child creation, not live propagation of later parent rule changes.

Custom engines (including subclasses of the default engine) receive child requests
through a small authorization view forwarding `authorize` and `aauthorize`.
Their constructor is never guessed and their custom decisions are preserved.
This view does not isolate internal state or reinterpret custom mode semantics.
For an engine with agent-local state, explicitly construct a child instance:

```python
agent.with_multi_agent(
    child_permission_factory=lambda parent_policy, runtime: MyPolicy(
        policy_directory=policy_directory,
    ),
)
```

The factory must return a separate BasePermissionEngine. Agent injects the child
Runtime through its normal `with_permissions` path. Custom module instances passed
to `with_multi_agent(module)` own their assembly; these factory arguments configure
only the default implementation.

EasyAgent intentionally keeps `dontAsk` mapped to `full_access`; this differs from
Claude Code. Explicit deny rules and Plan restrictions still apply.

Observability keeps its separate child subscription contract: default children
share the backing store but not invocation aggregation state; closing the child
does not close the parent's sink. Plan children likewise retain independent state.

## Context and Executor

Context exposes separate input construction, usage, compaction and validation
stages. `validation_epoch(compaction)` determines the generation used to validate
a candidate input; Executor must not infer how a Context advances its versions.
`get_usage()` returns the typed `ContextUsageReport`, including for custom Context
implementations using the base convenience method. Provider response objects remain
opaque at conversion boundaries, while history, prompt and result contracts are typed.

## Optional session participation

Session persistence uses the structural `SessionState` protocol: `export_state()`
and `restore_state(state)`. It does not impose an `AgentModule` superclass. State
is a serializable dictionary or `None`; restoration returns `None`, a typed report
dictionary, or `ComponentRestoreReport`.

Optional validation, reset and fork protocols are defined in `session/state.py`.
A participant can be registered without changing Agent or Executor:

```python
agent.register_session_state("counter", counter, required=True)
```

Restore applies data to already assembled instances. Invalid restore results are
rejected and rollback is attempted for every changed participant; incomplete
rollback is reported, not silently swallowed. External stores and processes are
not reconstructed from class names.

## Regression checks

`test/test_composition_boundaries.py` exercises custom implementations together,
including child Plan isolation, non-copyable authorization resources, child event
subscriptions, custom Task/CodeIntel/Worktree modules, state restore, and Context
dispatch through sync, async and both streaming paths.
`test/test_child_tool_bindings.py` additionally covers shared tool definitions,
registry isolation, function/async tools, child workspace and shell bindings,
borrowed MCP connections, explicit factories, post-assembly filtering, permission
descriptors and the preserved dontAsk semantics. These tests make no paid model
calls.
