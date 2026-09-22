# Multi-Agent Module Guide

Multi-Agent is an optional EasyAgent module. The framework defines how an implementation contributes tools, prompt blocks, request-time context, session state, and lifecycle cleanup. It does not require every implementation to adopt subagents, teams, mailboxes, actors, or a graph.

## Framework contract

Custom implementations extend `BaseMultiAgent`:

```python
from easyagent.multiagent import BaseMultiAgent

class MyMultiAgent(BaseMultiAgent):
    def bind(self, *, tool_registry, execution_context, event_bus, metamessages):
        self.context = execution_context

    def get_tools(self):
        return (MyDelegateTool(...),)

    def get_prompt_blocks(self, context):
        return ()
```

Install it with:

```python
agent.with_multi_agent(MyMultiAgent())
```

`with_multi_agent()` passes explicit framework facilities, registers returned tools, registers the prompt provider, includes the module in session persistence, and closes it with the Agent. The implementation never receives the parent Agent as a service locator.

`before_context_build()` is an optional lifecycle method. It can publish dynamic collaboration input through `MetaMessageManager` before the Context module assembles the next request.

## Built-in Claude Code-style implementation

Calling `agent.with_multi_agent()` without an argument installs `ClaudeCodeMultiAgent`. Its implementation lives entirely under `multiagent/default/`:

```text
multiagent/
├── base.py
└── default/
    ├── module.py
    ├── definitions.py
    ├── factory.py
    ├── agents/
    ├── teams/
    ├── subagents.py
    └── tools/
```

The default implementation provides Agent handles, foreground/background execution, Team state, automatic teammate-message delivery, shared Task integration, optional worktree isolation, and structured restore reports. Its model-facing control tools include `Agent`, `AgentGet`, `AgentList`, `AgentWait`, `AgentStop`, `SendMessage`, `TeamCreate`, and `TeamDelete`.

`AgentDefinitionRegistry` controls available `subagent_type` values. A definition can select a model, add child instructions, allow or deny tools, and choose a permission mode. Built-in definitions include `general-purpose`, `Explore`, and `Plan`.

```python
agent.with_multi_agent(
    workspace_root=".",
    storage_dir=".easyagent/agents",
    max_background_tasks=4,
)
```

Team messages are delivered automatically as `<teammate-message>` MetaMessages before Context assembly. Models use `SendMessage` to communicate; there are no model-facing mailbox polling tools.

## Boundary rule

`BaseMultiAgent` contains only integration points common to replaceable implementations. Agent definitions, teams, mailboxes, background threads, worktrees, and Claude-aligned prompts belong to `ClaudeCodeMultiAgent`. A custom actor, swarm, remote-worker, or single-delegate implementation can replace it without emulating those concepts.
