# Session Restore Persistence Guide

Session 模块解决的是“Agent 运行到一半，或者对话结束后，我要如何保存状态并在之后恢复”。  
它不是单纯的聊天记录数据库，而是**会话快照 + 消息历史 + 恢复报告**的组合。

相关文档：

- [Agent Guide](./agent_guide.md)
- [Memory System Guide](./memory_system_guide.md)
- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)
- [Context And Compaction Guide](./context_and_compaction_guide.md)

## 1. Session 模块包含什么

### `SessionStore`

保存 session 元数据和 agent snapshot。  
典型字段：

- `session_id`
- `agent_type`
- `agent_name`
- `snapshot`
- `metadata`
- `created_at`
- `updated_at`
- `last_accessed_at`
- `expires_at`

### `ConversationStore`

保存对话消息序列。  
消息表至少包含：

- `session_id`
- `position`
- `role`
- `content`
- `time`
- `metadata`
- `tool_call_id`
- `name`
- `raw_message`

### `SessionRestoreReport`

恢复不是简单的成功/失败。  
EasyAgent 会生成结构化恢复报告，告诉你：

- 哪些组件恢复成功
- 哪些组件部分恢复
- 哪些工具/skill 缺失
- execution context 是否恢复

### `ComponentRestoreReport`

每个子组件自己的恢复报告。

### `RestoreIssue`

具体问题项。至少包含：

- `component`
- `code`
- `message`
- `severity`
- `metadata`

## 2. Session 保存的内容边界

### 通常会保存

- agent 的基础快照
- 历史消息
- replay 相关状态
- execution context 的结构化信息
- cache / restore 所需元信息
- 某些 registry / runtime 的可恢复状态摘要

### 不保证自动恢复

以下内容通常仍建议由产品层在 `load_session(...)` 时显式补回：

- 外部数据库连接
- 真实网络连接
- 某些 MCP client 实例
- 某些本地文件句柄 / 进程句柄
- 不是纯数据的复杂依赖对象

## 3. 为什么需要恢复报告

因为在真实产品里，“部分恢复成功”比“恢复失败”更常见。

例如：

- session 历史恢复了
- prompt blocks 恢复了
- execution context 恢复了
- 但某个 skill 现在不在 registry 里
- 或某个 runtime surface 缺了

如果只返回 `True/False`，上层根本不知道还缺什么。

## 4. 保存流程

一个典型的保存流程是：

1. agent 导出 snapshot
2. `SessionStore.create_or_update_session(...)`
3. `ConversationStore.replace_messages(...)`
4. 更新 session metadata / last_accessed_at

产品层往往还会额外保存：

- UI 状态
- 当前 tab / project
- 用户层偏好

这些不一定要放进 EasyAgent 的 session snapshot 主体，但应与你的产品存储层一起设计。

## 5. 恢复流程

一个典型的恢复流程是：

1. 从 `SessionStore` 读取 session snapshot
2. 从 `ConversationStore` 读出消息历史
3. 构造新的 Agent 实例
4. 恢复 replay / history / context state
5. 恢复 execution context
6. 检查 tool / skill / runtime 是否齐全
7. 生成 `SessionRestoreReport`

## 6. 和 Agent 的集成方式

产品层通常会这样接：

1. 构造 `SessionStore` 和 `ConversationStore`
2. agent 执行一段时间后调用保存
3. 下次启动时再构造 agent，然后调用 load

这里的关键原则是：

- 先构造 agent
- 再恢复 snapshot

而不是指望单个静态函数替你自动重建所有依赖。

## 7. `SessionRestoreReport` 应如何使用

推荐上层这样处理：

- `status == restored`
  - 直接恢复
- `status == degraded`
  - 恢复成功，但提示用户哪些组件降级
- `status == failed`
  - 中止继续恢复，要求重新初始化

同时把：

- `missing_tools`
- `missing_skills`
- `components`
- `issues`

展示到调试面板或日志中。

## 8. Session 和 Memory / Runtime / Context 的关系

### Session vs Memory

- session 保存当前运行的会话状态
- memory 保存长期可检索知识

两者有关，但不是一回事。

### Session vs Runtime

runtime 的某些状态可以进入 snapshot，但后台任务、进程、网络连接等不应该被想象成“完全自动恢复”。

### Session vs Context

context builder 和 compaction 可能改变 replay history 形态，因此 session 恢复后需要重新确认：

- summary 是否存在
- compaction metadata 是否存在
- replay history 是否仍可继续使用

## 9. 推荐实践

### 轻量产品

- 只保存对话历史和基础 snapshot

### 工程型 Agent

- 再加 execution context、permission context、tool/runtime 元数据

### 多 Agent 系统

- 把 session 恢复报告和 runtime 恢复报告分层展示

## 10. 常见坑

### 以为恢复 = 所有外部依赖自动重连

不是。  
真正的连接类对象一般还是要在产品层显式补回。

### 只保存消息，不保存 snapshot

这样恢复后你只能拿到聊天记录，拿不到足够的运行状态。

### 忽略恢复报告

这会让“恢复看似成功但其实缺组件”的问题很难排查。
