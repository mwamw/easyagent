# Framework API

这份文档定义 EasyAgent 当前推荐的公共 SDK 边界，也说明哪些模块适合上层产品直接依赖。

相关文档：

- [README](../README.md)
- [Config Reference](./config_reference.md)
- [Agent Guide](./agent_guide.md)
- [Tool System Guide](./tool_system_guide.md)
- [Prompt System Guide](./prompt_system_guide.md)

## 1. 公共入口原则

新项目优先从 `easyagent` 或 `easyagent.*` 导入。
不要把 `agent/`、`core/`、`Tool/`、`runtime/` 当作稳定 SDK 边界。

原因：

- `easyagent.*` 是对外承诺的模块层
- 内部目录更偏实现组织，后续可能继续演进
- 上层产品应依赖稳定包装层，而不是内部文件结构

## 2. 顶层导出

最常见的顶层导出：

```python
from easyagent import BasicAgent, EasyLLM, Config, ToolRegistry
```

顶层适合快速原型，但做产品时建议逐步切到子模块导入，代码会更清晰。

## 3. 公共子模块

### `easyagent.agents`

导出：

- `BaseAgent`
- `BasicAgent`
- `ConversationalAgent`
- `PlanningAgent`
- `ReactAgent`
- `StructuredOutputAgent`

用途：

- 组装 agent
- 替换默认运行组件
- 做产品自己的 agent 子类

详见：

- [Agent Guide](./agent_guide.md)

### `easyagent.llms`

导出：

- `EasyLLM`

用途：

- 初始化 provider
- 配置模型、base_url、api_key、reasoning
- 统一 tool-calling / streaming 接口

详见：

- [LLM Provider Guide](./llm_provider_guide.md)

### `easyagent.config`

导出：

- `Config`

用途：

- 控制 tool schema mode
- cache policy
- worktree / shell / roots
- compaction、subagent、reasoning 持久化等

详见：

- [Config Reference](./config_reference.md)

### `easyagent.tools`

导出：

- `Tool`
- `ToolRegistry`
- `ToolResult`
- `ToolSpec`
- `ToolConflictPolicy`
- `ToolVisibilityScope`
- 内置 `register_*` helper

用途：

- 自定义工具
- 注册内置工具
- 批量装配 code agent 工具集合

详见：

- [Tool System Guide](./tool_system_guide.md)
- [Builtin Tools Catalog](./builtin_tools_catalog.md)
- [Tool Authoring Guide](./tool_authoring_guide.md)

### `easyagent.permissions`

导出：

- `PermissionContext`
- `PermissionRule`
- `PermissionDecision`
- `PermissionBehavior`
- `PermissionMode`
- `PermissionEngine`
- `PermissionStore`
- `RiskCategory`

详见：

- [Permissions Guide](./permissions_guide.md)

### `easyagent.callbacks`

导出：

- `BaseCallback`
- `CallbackEvent`
- `CallbackManager`
- `StreamingCallback`
- `LoggingCallback`
- `MetricsCallback`

详见：

- [Callbacks And Streaming Guide](./callbacks_and_streaming_guide.md)

### `easyagent.hooks`

导出：

- `BaseHook`
- `HookAction`
- `HookDecision`
- `HookExecutionResult`
- `HookManager`

详见：

- [Hooks And Guardrails Guide](./hooks_and_guardrails_guide.md)

### `easyagent.prompting`

导出：

- `PromptBlock`
- `SystemPromptTemplate`
- `BasePromptComposer`
- `DefaultPromptComposer`

详见：

- [Prompt System Guide](./prompt_system_guide.md)
- [Prompt Composer Guide](./prompt_composer_guide.md)

### `easyagent.reminders`

导出：

- `RuntimeReminder`
- `BaseRuntimeReminderSource`
- `StaticRuntimeReminderSource`

详见：

- [Runtime Reminders Guide](./runtime_reminders_guide.md)

### `easyagent.skills`

导出：

- `BaseSkill`
- `SkillConfig`
- `SkillManifest`
- `SkillRegistry`
- `SkillManager`
- `SkillTool`
- `LoadSkillTool`
- `UnloadSkillTool`
- `MetaSkill`
- `FolderSkill` / `MarkdownSkill` / `YAMLSkill`
- `MCPSkill` / `MCPPromptSkill`

详见：

- [Skill System Guide](./skill_system_guide.md)

### `easyagent.context`

导出：

- `ContextManager`
- `ContextBuilder`
- `BaseContextSource`
- `TokenBudget`
- `TokenCounter`
- `BaseHistoryCompactor`
- `LLMHistoryCompactor`
- `RuleBasedHistoryCompactor`
- 多种 formatter / compressor

详见：

- [Context And Compaction Guide](./context_and_compaction_guide.md)

### `easyagent.memory`

导出：

- `MemoryManage`
- `MemoryConfig`
- `MemoryItem`
- `MemoryType`
- `WorkingMemory`
- `BaseMemory`

详见：

- [Memory System Guide](./memory_system_guide.md)

### `easyagent.session`

导出：

- `SessionStore`
- `ConversationStore`
- `SessionRestoreReport`
- `ComponentRestoreReport`
- `RestoreIssue`

详见：

- [Session Restore Persistence Guide](./session_restore_persistence_guide.md)

### `easyagent.runtime`

导出：

- `AgentRuntimeManager`
- `AgentHandle`
- `BackgroundAgentHandle`
- `ExecutionContext`
- `MailboxMessage`
- `CompletionRecord`
- `TeamManager`
- `TeamHandle`

详见：

- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)

### `easyagent.tasks`

导出：

- `TaskService`
- `TaskRecord`
- `TaskStatus`
- `InMemoryTaskStore`
- `SQLiteTaskStore`

详见：

- [Tasks Guide](./tasks_guide.md)

### `easyagent.mcp`

导出：

- `MCPToolManager`
- `register_mcp_tools`
- `register_mcp_resource_hub_tools`
- `build_mcp_hub_resource_tools`
- `MCPClient`
- `MCPHub`
- `MCPRuntimeManager`
- `MCPConnectionManager`
- `MCPAuthConfig`
- `MCPPolicyContext`

详见：

- [MCP Guide](./mcp_guide.md)

### `easyagent.codeintel`

导出：

- `CodeIntelManager`
- `CodeIntelProvider`
- `LSPCodeIntelProvider`
- `WorkspaceCodeIntelCache`
- 位置、诊断、符号相关 query / result model

详见：

- [CodeIntel Guide](./codeintel_guide.md)

### `easyagent.observability`

导出：

- `BaseObservabilityRecorder`
- `InMemoryObservabilityRecorder`

详见：

- [Observability And Cache Guide](./observability_and_cache_guide.md)

### `easyagent.rag`

导出：

- `RAGPipeline`
- `DocumentLoader`
- `BaseChunker`
- `BaseEmbedding`
- `BaseRetriever`
- `BaseVectorStore`
- 多种 chunker / retriever / vectorstore / query transformer

详见：

- [RAG Guide](./rag_guide.md)

### `easyagent.worktree`

导出：

- `WorktreeManager`
- `GitWorktreeInfo`
- `GitWorktreeSession`

详见：

- [Worktree Guide](./worktree_guide.md)

## 4. 一般不要直接依赖的内部目录

这些目录目前主要是内部实现，不建议新项目直接当 SDK 用：

- `agent/`
- `core/`
- `Tool/`
- `skill/`
- `runtime/`
- `context/`
- `memory/`
- `task/`
- `observability/`

只有在做框架二次开发，或你明确知道要替换内部实现组件时，才应直接引用这些路径。

## 5. 一个推荐的产品骨架

```python
from easyagent import BasicAgent, Config, EasyLLM
from easyagent.tools import ToolRegistry, register_filesystem_tools
from easyagent.permissions import PermissionContext, PermissionEngine
from easyagent.callbacks import CallbackManager
from easyagent.observability import InMemoryObservabilityRecorder

llm = EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="qwen3.5-9b")
config = Config(tool_schema_mode="deferred")
registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root=".")

agent = BasicAgent(
    name="product-agent",
    llm=llm,
    config=config,
    enable_tool=True,
    tool_registry=registry,
    permission_engine=PermissionEngine(),
    permission_context=PermissionContext(),
    callback_manager=CallbackManager(),
    observability_recorder=InMemoryObservabilityRecorder(),
)
```

## 6. 阅读顺序建议

推荐顺序：

1. [README](../README.md)
2. [Config Reference](./config_reference.md)
3. [Agent Guide](./agent_guide.md)
4. 根据你的目标继续看 Tool / Prompt / Runtime / Session / Memory / MCP
