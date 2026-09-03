# Framework API

新项目应从 `easyagent` 或 `easyagent.*` 导入。内部的 `core/`、`agent/`、`Tool/`、`runtime/` 等目录不是产品侧稳定入口。

## 1. Agent 与执行

`easyagent` 和 `easyagent.agents` 导出：

- `BaseAgent`
- `BasicAgent`
- `ConversationHistory`
- `BaseAgentExecutor`
- `DefaultAgentExecutor`
- `AgentExecutionServices`
- `AgentInvocationState`
- `AgentInvocationPhase`

框架只维护 BaseAgent/BasicAgent，不导出 specialized agent。

## 2. 模型和配置

`easyagent.llms`：

- `EasyLLM`

`easyagent.config`：

- `Config`

EasyLLM 统一 provider 请求、canonical codec、tool calling、streaming 和 usage 提取。Agent 流接口与 LLM 原始流接口是不同层级。

## 3. Prompt 与 MetaMessage

`easyagent.prompting`：

- `BaseSystemPromptComposer`
- `SystemPromptComposer`
- `PromptBuildContext`
- `PromptBlock`
- `PromptPlacement`
- `SystemPromptTemplate`

`easyagent.metamessages`：

- `BaseMetaMessageManager`
- `MetaMessageManager`
- `MetaMessage`
- `MetaMessageContext`
- `MetaMessageLifecycle`
- `MetaMessageEvent`

Prompt 管理 system/system reminder；MetaMessage 管理 history 尾部的运行时注入。

## 4. Tool、Permission、Hook、Callback

`easyagent.tools`：

- `Tool`
- `ToolSpec`
- `ToolResult`
- `ToolRegistry`
- `ToolConflictPolicy`
- `ToolSideEffectLevel`
- `ToolVisibilityScope`
- builtin `register_*` helpers

`easyagent.permissions`：

- `PermissionEngine`
- `PermissionContext`
- `PermissionStore`
- `PermissionRule`
- `PermissionDecision`
- `PermissionBehavior`
- `PermissionMode`
- `RiskCategory`

`easyagent.hooks`：

- `BaseHook`
- `HookManager`
- `HookDecision`
- `HookAction`
- `HookExecutionResult`

`easyagent.callbacks`：

- `BaseCallback`
- `CallbackManager`
- `CallbackEvent`
- `LoggingCallback`
- `MetricsCallback`
- `StreamingCallback`

Permission 决定能否执行；Hook 可以阻断或改写；Callback 只观察 RuntimeEvent。

## 5. Runtime 与协作

`easyagent.runtime`：

- `ExecutionContext`
- `RuntimeEvent`
- `RuntimeEventType`
- `RuntimeEventBus`
- `AgentStreamEvent`
- `AgentStreamEventType`
- `BaseMultiAgentRuntime`
- `MultiAgentRuntime`
- `AgentRuntimeManager`
- `AgentHandle`
- `BackgroundAgentHandle`
- `MailboxMessage`
- `CompletionRecord`
- `TeamManager`
- `TeamHandle`

产品通常只需调用 `agent.with_multi_agent()`；需要替换运行策略时再直接使用 manager/handle 类型。

## 6. Plan、Task、Skill、Context

`easyagent.plans`：

- `BasePlanMode`
- `PlanModeManager`
- `PlanModeConfig`
- `PlanModeState`
- `ExecutionMode`

`easyagent.tasks`：

- `TaskService`
- `TaskRecord`
- `TaskStatus`
- `BaseTaskStore`
- `InMemoryTaskStore`
- `SQLiteTaskStore`

`easyagent.skills`：

- `SkillManifest`
- `SkillManager`
- `SkillTool`
- directory discovery and lazy `SKILL.md` loading

`easyagent.context`：

- `ContextManager`
- `ContextBuilder`
- `BaseContextSource`
- `ContextItem`
- `TokenBudget`
- `TokenCounter`
- history compactor、compressor 和 formatter 实现

## 7. Observability 与 Training

`easyagent.observability`：

- `BaseObservabilityManager`
- `ObservabilityManager`
- `BaseObservabilityStore`
- `InMemoryObservabilityStore`
- `SQLiteObservabilityStore`
- `AgentInvoke`
- `LLMInvoke`
- 对应 stats model

`easyagent.training`：

- `TrainingExporter`
- `TrainingDataFilter`
- `SuccessfulAgentInvokeFilter`
- `TrainingDataFormat`
- `TrainingExportReport`

Training 只消费 observability store，不是 Agent 内部执行模块。

## 8. CodeIntel、MCP、Worktree

`easyagent.codeintel` 导出 provider、manager、LSP client、query/result model 和 workspace cache。

`easyagent.mcp` 导出 MCP client/hub/runtime/tool manager、连接管理、auth、cache、policy 和 capability snapshot。

`easyagent.worktree` 导出 `WorktreeManager`、`GitWorktreeInfo` 和 `GitWorktreeSession`。

这些能力均通过对应 `agent.with_*` 安装。

## 9. Session

`easyagent.session`：

- `SessionStore`
- `ConversationStore`
- `SessionRestoreReport`
- `ComponentRestoreReport`
- `RestoreIssue`

Snapshot 显式记录模块 envelope。恢复不会扫描工具名推断模块。

## 10. 推荐装配

```python
from easyagent import BasicAgent, Config, EasyLLM, PlanModeConfig
from easyagent.context import ContextManager
from easyagent.tools import ToolRegistry, register_filesystem_tools

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root=".")

agent = (
    BasicAgent("product-agent", llm, config=Config(workspace_root="."))
    .with_tool(registry)
    .with_context(ContextManager(max_tokens=16000))
    .with_plan(config=PlanModeConfig(register_tools=True))
    .with_observability(path=".easyagent/observability.sqlite3")
)
```

完整真实过程见 [`example_modular_agent_framework.py`](../example/example_modular_agent_framework.py)。
