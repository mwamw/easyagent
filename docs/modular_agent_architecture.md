# EasyAgent 模块化架构

本文描述当前 EasyAgent 的正式架构和公共使用方式。它以实际代码为准，不延续旧版 specialized agent、构造器注入和私有执行组件的接口。

## 1. 架构目标

EasyAgent 是通用 Agent 框架，不是一个固定的 Code Agent 产品。框架提供可直接使用的工业实现，同时允许用户按模块替换实现。

当前原则如下：

- 只维护 `BaseAgent` 和 `BasicAgent` 两种 Agent 类型。
- 不定义统一的 `AgentModule` 契约；每个模块按自己的职责设计接口。
- 模块可以依赖真正需要的模块，例如 Plan 依赖 Permission 和 MetaMessage，MultiAgent 依赖 Tool。
- 用户通过 `with_*` 显式安装能力，不需要理解 Agent 内部执行器的接线细节。
- 扩展边界采用名义类型。自定义实现应继承对应模块的基类，而不是依赖运行时 duck typing。
- Code Agent 是必须支持的场景，但关闭 CodeIntel、Worktree、Task 和 MultiAgent 后，BasicAgent 仍是完整的通用 Agent。

## 2. 当前结构

```text
BasicAgent
  -> BaseAgent
     -> ConversationHistory
     -> SystemPromptComposer
     -> MetaMessageManager
     -> SkillManager
     -> PermissionEngine / PermissionContext
     -> HookManager / CallbackManager
     -> RuntimeEventBus
  -> DefaultAgentExecutor

optional modules
  -> ToolRegistry
  -> ContextManager
  -> PlanModeManager
  -> TaskService
  -> ObservabilityManager
  -> MultiAgentRuntime
  -> MemoryManage
  -> MCPToolManager
  -> CodeIntelManager
  -> WorktreeManager
```

`BaseAgent` 负责身份、固定状态和模块装配。`DefaultAgentExecutor` 负责一次调用的状态机和 tool loop。模块之间通过明确依赖、`ExecutionContext` 和 `RuntimeEventBus` 协作，不再通过多个执行辅助类相互读取 Agent 私有字段。

## 3. BasicAgent 的最小形态

构造器只接受五个参数：

```python
BasicAgent(
    name,
    llm,
    system_prompt=None,
    description=None,
    config=None,
)
```

默认存在的轻量能力：

| 能力 | 默认实现 | 说明 |
| --- | --- | --- |
| History | `ConversationHistory` | 保存 provider-neutral canonical history，并维护当前 provider replay |
| Prompt | `SystemPromptComposer` | 生成 native system 和 request-level system reminder |
| MetaMessage | `MetaMessageManager` | 管理运行时消息注入与回收 |
| Skill | 空 `SkillManager` | 未注册 Skill 时没有额外成本 |
| Permission | `PermissionEngine` + `PermissionContext` | Tool 未安装时不参与执行 |
| Hook | 默认 guardrail `HookManager` | 提供执行前后策略边界 |
| Callback | `CallbackManager` | 订阅统一 runtime 事件 |
| Event | `RuntimeEventBus` | 保存本 Agent 的结构化执行事件 |
| Executor | `DefaultAgentExecutor` | 提供四种正式调用方式 |

默认不安装的重能力：

- Tool
- Context / compaction
- Plan
- Task
- Observability
- MultiAgent
- Memory
- MCP
- CodeIntel
- Worktree

最小通用 Agent：

```python
from easyagent import BasicAgent, EasyLLM

agent = BasicAgent(name="assistant", llm=EasyLLM(...))
answer = agent.invoke("解释这个概念。")
```

## 4. 模块装配接口

正式装配接口如下：

| 接口 | 安装内容 | 自动依赖 |
| --- | --- | --- |
| `with_prompt(composer)` | 自定义系统提示词组装器 | 无 |
| `with_skill(*directories, manager=None)` | 目录式 Skill 与 Agent 私有 SkillManager | Tool |
| `with_tool(registry=None)` | ToolRegistry 和 tool loop | 无 |
| `with_context(manager)` | 外部上下文与 compaction | 无 |
| `with_permissions(engine=None, context=None)` | 权限策略 | 无 |
| `with_hooks(manager)` | 可阻断、可修改的执行 hook | 无 |
| `with_callbacks(manager)` | 非阻断运行通知 | RuntimeEventBus 已提供 |
| `with_plan(plan=None, config=None)` | Plan 状态和进入/退出规则 | 配置 Plan tools 时安装 Tool |
| `with_task_service(service)` | 结构化 Task 和 task tools | Tool |
| `with_observability(...)` | AgentInvoke / LLMInvoke 数据记录 | RuntimeEventBus |
| `with_multi_agent(...)` | subagent、team、mailbox 和控制工具 | Tool |
| `with_memory(manager)` | Memory context 和 memory tools | Context，必要时 Tool |
| `with_mcp(...)` | MCP lifecycle 和远程工具 | Tool |
| `with_codeintel(...)` | CodeIntel provider 和查询工具 | Tool |
| `with_worktree(...)` | Git worktree 生命周期和工具 | Tool |
| `with_executor(executor)` | 替换 Agent 执行策略 | 无 |
| `with_interruptions(controller)` | 替换工具确认中断存储 | Tool 执行时使用 |

模块的自动依赖只在能力确实无法独立工作时发生。调用 `with_skill("./skills")` 或 `with_multi_agent()` 本身就是显式启用该能力，因此它们可以补装 Tool；创建一个最小 BasicAgent 不会隐式安装这些重模块。

## 5. System Prompt 与 MetaMessage

两者都是给模型的指令，但生命周期和传输位置不同。

### 5.1 System Prompt

`PromptBlock` 的 `placement` 只有两个值：

- `system`：进入 provider 原生 system 字段。
- `system_reminder`：包装成 `<system-reminder>`，只加入当前请求的 prepended replay，不写入 canonical history。

最简单的自定义方式：

```python
from easyagent import PromptBlock, SystemPromptComposer

composer = SystemPromptComposer(
    blocks=[
        PromptBlock("identity", "你是游戏开发 Agent。", placement="system"),
        PromptBlock(
            "workspace",
            lambda ctx: f"当前工作区：{ctx.execution_context.workspace_root}",
            placement="system_reminder",
        ),
    ]
)
agent.with_prompt(composer)
```

callable 接收 `PromptBuildContext`，不接收 Agent 本身。用户不需要实现请求编译、provider 格式或默认 block 拼装。

### 5.2 MetaMessage

MetaMessage 是 Agent 内建运行时基础设施，不是通过 `with_*` 安装的用户功能模块。它把模块在运行事件中产生的临时上下文写入 history 末尾，供 Skill、Plan、mailbox、工具上下文和用户自定义模块复用。

生命周期：

- `PERMANENT`：注入后永久留在历史中。
- `INVOCATION`：只在一次 Agent invoke 中存在，invoke 结束自动回收。
- `REQUEST`：只在一次 LLM request 中存在，请求成功后回收。
- `CONDITIONAL`：条件为真时注入，条件变为假时自动回收。

模块应在自己的触发逻辑中调用 `agent.emit_metamessage(...)`，而不是要求 Agent 使用者直接注册 MetaMessage。永久静态规则属于 SystemPromptComposer；运行时消息的触发和生命周期由具体模块负责。

Plan、按需 Skill、mailbox 和工具临时上下文都复用这套注入机制：

- Plan enter 注入永久的 plan 规则。
- Plan exit 注入永久的 execute 规则，历史保留完整模式变化。
- Skill 在加载时发出 invocation MetaMessage，调用结束自动回收。
- MultiAgent 每次 request 前同步 mailbox，将未消费消息注入 history。
- `ToolResult.ephemeral_context` 转成 invocation MetaMessage，不污染稳定 system prompt。

## 6. 执行器与调用状态机

Agent 只公开四种调用方式：

```python
agent.invoke(query)
await agent.ainvoke(query)
for event in agent.stream(query): ...
async for event in agent.astream(query): ...
```

是否进入 tool loop 由已安装的 ToolRegistry 和当前可见工具决定，不再提供 `invoke_with_tool`、`stream_invoke` 等平行 API。

一次调用由 `AgentInvocationState` 跟踪，主要阶段是 preparing、LLM、tool、completed、failed 和 interrupted。`DefaultAgentExecutor` 负责：

1. 启动 MetaMessage invocation。
2. 同步 MultiAgent mailbox。
3. 执行 Skill before hook。
4. 构建 Prompt 和 Context request input。
5. 必要时运行 compaction precheck。
6. 调用 LLM，并解析 canonical output。
7. 执行 permission、hook、tool 和 interruption 流程。
8. 继续下一轮，直到得到 final response。
9. 发布完成、失败或中断事件并回收临时状态。

用户替换执行策略时继承 `BaseAgentExecutor`，无需修改 BasicAgent。

## 7. 统一 Runtime Event

`RuntimeEventBus` 是模块间的稳定观测边界。事件包括：

- `agent.invoke.started/completed/failed/interrupted`
- `llm.invoke.started/completed/failed`
- `tool.invoke.started/completed/failed`
- `history.compacted`
- `agent.stream.event`

事件统一包含：

- `event_id`
- `type`
- `agent_id`
- `invocation_id`
- `sequence`
- `timestamp`
- `data`

EventBus 会保存历史、隔离 subscriber 异常，并随 session snapshot 恢复 sequence。Callback 和 Observability 都通过订阅事件工作，不再被 executor 直接调用。

流式接口产出 `AgentStreamEvent`：

- `text_delta`
- `reasoning_delta`
- `tool_call`
- `tool_result`
- `final`
- `error`

每个事件都有 `invocation_id`、严格递增的 `sequence`、`content` 和结构化 `data`，可通过 `to_dict()` 直接传给 CLI、TUI、WebSocket 或日志层。

## 8. MultiAgent 是独立模块

`MultiAgentRuntime` 是多智能体能力的 façade，内部拥有：

- `AgentRuntimeManager`
- `TeamManager`
- mailbox
- completion records
- Agent/Team/Message 控制工具
- session export/restore
- shutdown

安装：

```python
agent.with_multi_agent(workspace_root=".")
```

它会注册以下工具：

- `Agent`
- `AgentGet`
- `AgentList`
- `AgentWait`
- `AgentStop`
- `SendMessage`
- `MailboxRead`
- `MailboxAck`
- `TeamCreate`
- `TeamDelete`

Agent 相关工具都返回结构化 handle 信息，包括 `agentId`、`status`、`outputFile`、`executionContext`、background 状态和错误信息。后台 Agent 的完整输出写入 `outputFile`，调用方不需要从一句“已启动”文本猜测结果位置。

### 8.1 mailbox 的真实读取链路

1. `SendMessage` 调用 `AgentRuntimeManager.send_message()`，消息进入目标 handle 的 mailbox。
2. 子 Agent 每次构建 LLM request 前调用 `MultiAgentRuntime.sync_mailbox()`。
3. 未消费消息被标记为 delivered，并通过永久 MetaMessage 注入子 Agent history。
4. 模型可以立即从当前 request 看到消息。
5. `MailboxRead` 提供结构化查看；`MailboxAck` 将消息标记为 consumed。
6. dedup key 保证同一消息不会重复写入 history。

这不是把 mailbox 文本拼进 system prompt，也不要求子 Agent 轮询私有字段。

## 9. Observability 与 Training

Observability 是可选 Agent 模块：

```python
agent.with_observability(path=".easyagent/observability.sqlite3")
```

它只消费统一 runtime event，并保存两层数据：

### `LLMInvoke`

- 本次 LLM 的完整 canonical input
- provider-neutral canonical output
- tools schema 和 options
- input/output/total/cache/reasoning token
- duration、success、error
- metadata

### `AgentInvoke`

- 用户 query
- `list[LLMInvoke]`
- 本次 Agent 的完整 canonical trace 和 output
- tool 次数、token、duration、success、error
- parent invoke，用于多 Agent trajectory
- metadata

Training 不写入 Agent，也不重新采集运行数据。它只消费 Observability store：

```python
from easyagent import TrainingDataFormat, TrainingExporter

exporter = TrainingExporter.from_agent(agent)
report = exporter.export(
    ".easyagent/training",
    formats=[
        TrainingDataFormat.STEP_SFT,
        TrainingDataFormat.TRACE_SFT,
        TrainingDataFormat.AGENTIC_ROLLOUT,
    ],
)
```

默认 `SuccessfulAgentInvokeFilter` 只导出已结束且成功的 trace。自定义清洗逻辑继承 `TrainingDataFilter`，实现 `accept()` 和可选的 `transform()`。

## 10. Session 与恢复

Session snapshot 显式保存 Agent 状态和已安装模块，不扫描 ToolRegistry 猜测模块。当前状态至少包括：

- identity、config、reasoning
- canonical history 和 provider replay metadata
- prompt、metamessage、skill
- permission、plan、task
- execution context
- interruptions
- observability
- multi-agent runtime/team/mailbox
- MCP、CodeIntel、Worktree
- runtime event history

恢复时：

- 标准实现可以自动恢复。
- 自定义实现必须通过对应 load 参数显式提供。
- 缺失工具、provider drift、不可恢复模块和降级项进入 `SessionRestoreReport`。
- runtime event history 恢复时不会重新通知 subscriber。

关闭顺序是 MultiAgent、MCP、CodeIntel、Worktree、Observability、LLM，返回结构化 close report；一个组件关闭失败不会阻止其他组件释放资源。

## 11. 自定义模块边界

当前正式扩展基类：

- `BaseSystemPromptComposer`
- `BaseMetaMessageManager`
- `BasePlanMode`
- `BaseObservabilityManager`
- `BaseMultiAgentRuntime`
- `BaseAgentExecutor`

框架不要求它们拥有统一的 config、manager、install 或 lifecycle 方法。每个接口只表达该能力真正需要的契约。

示例：替换 Prompt 只实现 `build(context)`；替换 Executor 必须实现四种 invoke API；替换 Observability 则实现 event bind/consume、`list()` 和 close。Training 只通过 `list()` 消费其标准 `AgentInvoke` 记录。自定义类通过对应 `with_*` 接入。

## 12. 本阶段框架变化

### 变化一：构造器从总装配器变为身份入口

旧形态把 Tool、Context、Memory、Runtime、Permission、Observability 和多个执行组件全部塞进构造器。现在构造器固定为五个字段，能力通过链式接口表达：

```python
agent = (
    BasicAgent("code-agent", llm, config=config)
    .with_tool(registry)
    .with_context(context_manager)
    .with_plan()
    .with_observability(path=trace_path)
    .with_multi_agent(workspace_root=workspace)
)
```

从代码上可以直接看出 Agent 启用了哪些能力。

### 变化二：执行组件收敛为 Executor

旧版多个 assembler、runner、loop、renderer 和 recorder 交叉持有状态。现在固定数据进入 `AgentExecutionServices`，执行策略集中在 `DefaultAgentExecutor`，流式展示交给上层消费结构化事件。

### 变化三：临时指令统一为 MetaMessage

Plan、Skill、mailbox 和 tool ephemeral context 不再各自维护 prompt 注入逻辑。它们只负责在正确事件发出 MetaMessage，manager 自动插入、去重和回收。

### 变化四：Trace 成为统一事件和训练事实源

Callback、Observability、session 和 training 不再维护不同 trace 模型。RuntimeEvent 是执行事实，Observability 把事件沉淀为 AgentInvoke/LLMInvoke，Training 再从这些记录导出语料。

### 变化五：多智能体从工具集合变为运行时模块

`MultiAgentRuntime` 统一安装 subagent、team 和 mailbox 控制面，并负责恢复和关闭。`Agent` 工具只做派发；handle、output file 和生命周期状态由 runtime 管理。

## 13. 已删除的旧接口

不提供兼容层：

- `ReactAgent`
- `PlanningAgent`
- `ConversationalAgent`
- `StructuredOutputAgent`
- `enable_tool` 等 BasicAgent 构造器模块参数
- `invoke_with_tool / ainvoke_with_tool`
- `stream_invoke / astream_invoke`
- 旧 invocation runner、tool loop、stream renderer、trace recorder 组件
- `MetaMessageManager(agent=...)`
- Prompt/MetaMessage context 中的 Agent 反向引用
- ToolRegistry camelCase 方法
- EasyLLM 的 `provide/resovle_*` 拼写错误属性和打印式兼容流接口

迁移规则很直接：构造参数改为 `with_*`；所有 Agent 调用改为四个正式 API；所有流式展示改为消费 `AgentStreamEvent`；所有运行记录改为消费 RuntimeEvent 或 Observability。

## 14. 真实示例

完整示例见：

- [`example/example_modular_agent_framework.py`](../example/example_modular_agent_framework.py)

示例使用真实 `EasyLLM`，安装 Prompt、MetaMessage、Tool、Context、Plan、Task、CodeIntel、Observability 和 MultiAgent，执行 plan/execute 两次调用，最后从真实 trace 导出三种训练数据。该文件只作为交付示例提供，框架测试不会执行它。
