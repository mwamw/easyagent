# Agent Guide

`BasicAgent` 是 EasyAgent 最常用的运行入口。  
它建立在 `BaseAgent` 之上，把模型、工具、上下文、技能、会话、权限、回调、Hook、Runtime 组装成一个可工作的 Agent。

相关文档：

- [Config Reference](./config_reference.md)
- [Tool System Guide](./tool_system_guide.md)
- [Prompt System Guide](./prompt_system_guide.md)
- [Session Restore Persistence Guide](./session_restore_persistence_guide.md)

## 1. Agent 的职责

一个 `BasicAgent` 主要负责：

- 接收 query
- 生成 request frame
- 调模型
- 处理中间 tool calls
- 维护历史
- 在需要时中断给上层做确认
- 记录 callback / hook / observability
- 按配置接入 skill、context、memory、session、runtime

## 2. 最小用法

```python
from easyagent import BasicAgent, EasyLLM

llm = EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="x", model="qwen3.5-9b")
agent = BasicAgent(name="assistant", llm=llm)
result = agent.invoke("解释一下当前仓库做什么。")
```

## 3. `BasicAgent(...)` 常见初始化参数

### 基础字段

- `name`
  - agent 名称
- `llm`
  - `EasyLLM` 实例
- `system_prompt`
  - 直接追加或替换的系统提示词文本
- `description`
  - agent 描述，常用于 runtime / task / UI 展示

### 工具相关

- `enable_tool`
  - 是否启用 tool loop
- `tool_registry`
  - `ToolRegistry` 实例

### 配置与上下文

- `config`
  - `Config` 对象
- `memory_manage`
  - `MemoryManage` 实例
- `context_manager`
  - `ContextManager` 实例
- `history_via_context_manager`
  - 是否将历史构建职责交给 context manager

### 运行控制

- `callback_manager`
  - `CallbackManager`
- `skill_manager`
  - `SkillManager`
- `permission_engine`
  - `PermissionEngine`
- `permission_context`
  - `PermissionContext`
- `hook_manager`
  - `HookManager`
- `task_service`
  - `TaskService`
- `agent_runtime`
  - `AgentRuntimeManager`
- `team_manager`
  - `TeamManager`
- `execution_context`
  - `ExecutionContext`

### 调试与推理

- `verbose_thinking`
  - 是否输出更完整的 thinking 流
- `reasoning`
  - provider-specific reasoning / thinking 配置
- `trace_recorder`
  - trace 记录器
- `stream_renderer`
  - 流式输出渲染器

### 可替换内部组件

- `prompt_composer`
- `history_message_assembler`
- `runtime_skill_context_bridge`
- `tool_interrupt_controller`
- `tool_loop_engine`
- `invocation_runner`

## 4. 一次 invoke 的主流程

1. 用户传入 query
2. Agent 收集 prompt blocks、runtime reminders、dynamic tail、skills、context
3. RequestCompiler 编译出 request frame
4. `EasyLLM` 发请求
5. 若模型返回 tool calls：
   - 进行权限判定
   - 可能触发确认中断
   - 执行工具
   - 把 tool result 追加回本轮
6. 继续下一轮直到产出最终回答
7. 记录 observability、callback、trace

## 5. 和其他模块怎么接

### 接 Tool

```python
agent = BasicAgent(
    name="code-agent",
    llm=llm,
    enable_tool=True,
    tool_registry=registry,
)
```

详见：

- [Tool System Guide](./tool_system_guide.md)

### 接 Prompt / Reminder

```python
agent.with_prompt_block(...)
agent.with_runtime_reminder(...)
agent.add_runtime_reminder_source(...)
```

详见：

- [Prompt System Guide](./prompt_system_guide.md)
- [Runtime Reminders Guide](./runtime_reminders_guide.md)

### 接 Skill

```python
agent.skill_manager = skill_manager
```

或在构造时传入 `skill_manager`。

详见：

- [Skill System Guide](./skill_system_guide.md)

### 接 Permission

```python
agent = BasicAgent(
    ...,
    permission_engine=PermissionEngine(),
    permission_context=PermissionContext(),
)
```

详见：

- [Permissions Guide](./permissions_guide.md)

### 接 Callback / Hook

```python
agent = BasicAgent(
    ...,
    callback_manager=CallbackManager(),
    hook_manager=HookManager(),
)
```

详见：

- [Callbacks And Streaming Guide](./callbacks_and_streaming_guide.md)
- [Hooks And Guardrails Guide](./hooks_and_guardrails_guide.md)

### 接 Session / Runtime / Task

这三类通常不一定直接在 `BasicAgent(...)` 初始阶段全部装上，但在产品级集成中很常见。

详见：

- [Session Restore Persistence Guide](./session_restore_persistence_guide.md)
- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)
- [Tasks Guide](./tasks_guide.md)

## 6. 什么时候需要替换内部组件

### `prompt_composer`

当你要彻底控制 system prompt 的块级结构时。

### `history_message_assembler`

当你要改变 replay history 的构造逻辑时。

### `runtime_skill_context_bridge`

当你要改变临时 skill 正文如何注入当前请求时。

### `tool_loop_engine`

当你要改变工具调用循环策略时。

### `invocation_runner`

当你要改变 invoke / stream / error handling 的顶层运行方式时。

## 7. 推荐的最小产品装配

对多数产品，推荐这样起步：

```python
BasicAgent(
    name="product-agent",
    llm=llm,
    config=config,
    enable_tool=True,
    tool_registry=registry,
    callback_manager=callback_manager,
    permission_engine=permission_engine,
    permission_context=permission_context,
    observability_recorder=observability_recorder,
)
```

然后按需再接：

- skill
- session
- runtime
- memory
- MCP
- codeintel

## 8. 常见误区

### 只改 `system_prompt` 就够了吗

通常不够。  
产品级 Agent 还应考虑：

- tool registry
- permission policy
- prompt composer
- runtime reminder
- callback / hook / observability

### `BasicAgent` 和 `BaseAgent` 该直接用哪个

大多数场景直接用 `BasicAgent`。  
只有在做框架级二次开发时，才会直接围绕 `BaseAgent` 搭建。
