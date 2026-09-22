# Agent Guide

EasyAgent 当前只维护 `BaseAgent` 和 `BasicAgent`。`BaseAgent` 定义稳定状态、模块装配和四种调用协议，`BasicAgent` 安装默认 executor，可以直接运行。

完整架构见 [Modular Agent Architecture](./modular_agent_architecture.md)。

## 1. 最小 Agent

```python
from easyagent import BasicAgent, EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
agent = BasicAgent(name="assistant", llm=llm)
print(agent.invoke("用一句话介绍当前项目。"))
```

构造器固定为：

```python
BasicAgent(name, llm, system_prompt=None, description=None, config=None)
```

Tool、Context、Plan、Task、Observability、MultiAgent、Memory、MCP、CodeIntel 和 Worktree 都不属于构造参数。

## 2. 四种调用方式

```python
answer = agent.invoke(query)
answer = await agent.ainvoke(query)

for event in agent.stream(query):
    consume(event)

async for event in agent.astream(query):
    consume(event)
```

`invoke/ainvoke` 返回最终字符串。`stream/astream` 产出 `AgentStreamEvent`，事件类型包括文本增量、推理增量、工具调用、工具结果、最终结果和错误。

工具是否可用由当前安装的 `ToolRegistry` 决定，不存在额外的 `*_with_tool` 调用方式。

## 3. 安装能力

### Tool

```python
from easyagent.tools import ToolRegistry, register_filesystem_tools

registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root=".")
agent.with_tool(registry)
```

### Prompt

```python
from easyagent import PromptBlock, SystemPromptComposer

agent.with_prompt(
    SystemPromptComposer(
        [PromptBlock("product", "你是产品专属 Agent。")]
    )
)
```

### MetaMessage runtime

MetaMessageManager 是 Agent 内建运行时基础设施，不通过 `with_*` 安装。Skill、Plan、mailbox 和自定义功能模块在事件触发后通过 `agent.emit_metamessage(...)` 注入临时上下文；普通 Agent 使用者应通过具体功能模块或 SystemPromptComposer 表达行为，而不是直接管理 MetaMessage。

### Context 和 compaction

```python
from easyagent.context import ContextManager

agent.with_context(ContextManager(max_tokens=16000))
```

### Permission 和 Hook

```python
from easyagent import HookManager, RuleBasedPermissionEngine

agent.with_permissions(RuleBasedPermissionEngine())
agent.with_hooks(HookManager())
```

权限是可选模块，`with_permissions()` 安装默认规则引擎。Hook 管线由每个 Agent 的 Runtime 固定创建，初始为空；`with_hooks()` 注册默认 guardrail，传入 HookManager 则将其中的 Hook 注册到既有管线。它不会替换 Runtime 的 HookManager。

### Plan

```python
from easyagent import PlanModeConfig

agent.with_plan(config=PlanModeConfig(register_tools=True))
agent.enter_plan_mode()
agent.exit_plan_mode()
```

Plan 通过 MetaMessage 记录进入和退出指令，并更新现有 AgentRuntime 的执行模式。
Agent 的 `build_authorization_context()` 将当前 `runtime_mode` 和审批状态传给权限引擎；
Executor 不自行解释 Plan 权限。默认引擎限制 Plan 下的写操作，自定义引擎可以实现自己的规则。

`agent.exit_plan_mode()` 和 `agent.plan.exit()` 只退出 Plan，不改变审批模式或权限规则。
要调整审批模式，应显式调用 `agent.set_permission_mode(...)`，而不是给退出 Plan 传参。

### Task

```python
from easyagent.tasks import InMemoryTaskStore, TaskService

agent.with_task_service(TaskService(InMemoryTaskStore()))
```

该操作会安装 Tool 并注册 `TaskCreate/Get/Update/List` 与 `TodoWrite`。

### Observability 和 Training

```python
from easyagent import TrainingExporter

agent.with_observability(path=".easyagent/observability.sqlite3")
agent.invoke("完成任务")
TrainingExporter.from_agent(agent).export(".easyagent/training")
```

### MultiAgent

```python
agent.with_multi_agent(
    workspace_root=".",
    storage_dir=".easyagent/agents",
)
```

该模块统一提供 subagent、team、mailbox 和对应控制工具。

### Code Agent 能力

```python
agent.with_codeintel()
agent.with_worktree()
```

CodeIntel 不可用时保留 FileRead/Grep/Glob fallback。Worktree 只应在 Git 仓库中安装。

### MCP 和 Memory

```python
agent.with_mcp(server_source=["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])
agent.with_memory(memory_manager)
```

两者会安装各自真正需要的 Tool/Context 依赖。

## 4. History 与 Trace

Agent history 由当前 Context 模块拥有。默认 Context 内部使用 provider-neutral canonical history：

- `agent.history`：`list[CanonicalMessage]`
- `agent.replay_history`：当前 provider 可直接发送的 replay
- `agent.get_history()`：可序列化 canonical dict
- `agent.get_trace_history()`：可序列化 RuntimeEvent dict
- `agent.clear_history()`：清空对话历史
- `agent.clear_trace_history()`：清空事件历史

切换模型使用 `agent.change_model(llm)`。canonical history 保持不变，replay 会按新 provider 重建。

## 5. 工具确认中断

当权限结果要求确认时，executor 抛出 `ToolConfirmationRequired`，不会先伪造 tool result。上层可以读取：

```python
pending = agent.get_pending_interruption()
```

用户批准后，宿主在自己的授权边界内执行动作并得到真实工具结果，再通过：

```python
agent.resolve_pending_interruption(
    content="真实工具结果",
    ephemeral_context={"approval": "granted"},
)
```

再发起下一次 invoke。BaseAgent 不提供平行的工具执行 API；正常工具执行只发生在 Executor 的 ReAct 流中。自定义中断存储通过 `with_interruptions()` 安装。

## 6. Session 与关闭

```python
agent.save_session("session-id")
restored = build_agent()  # 应用自己的装配函数
restored.restore_session("session-id")
print(restored.get_last_restore_report())

report = agent.close()
```

先装配需要的模块，再恢复其会话数据；持久化层不会创建模块实例。关闭返回结构化报告。

## 7. 自定义 Agent 行为

大多数产品不需要再创建新的 Agent 类型。优先替换具体模块：

- Prompt：继承 `BaseSystemPromptComposer`
- Plan：继承 `BasePlanMode`
- Executor：继承 `BaseAgentExecutor`
- Observability：继承 `BaseObservabilityManager`
- MultiAgent：继承 `BaseMultiAgent`

只有调用协议本身需要变化时才继承 `BaseAgent`；此时必须实现 `invoke`、`ainvoke`、`stream` 和 `astream`。

`MetaMessageManager` 和 `RuntimeEventBus` 是框架固定的 Runtime 基础设施，不是可替换模块。
模块可以发送 MetaMessage 或订阅 RuntimeEvent，但不能通过 `with_*` 替换其生命周期语义。

## 8. 已删除接口

`ReactAgent`、`PlanningAgent`、`ConversationalAgent` 和 `StructuredOutputAgent` 已删除。旧构造器模块参数、`stream_invoke`、`astream_invoke`、`invoke_with_tool` 及旧执行组件也不再提供兼容层。
