# Phase 1A: Permission / Task / Session 收口

这一步完成的是 `Phase 1A` 里最核心的一段：把权限规则、`TodoWrite`、会话恢复真正收敛到同一套内核状态里。

## 这一步之后，框架发生了什么变化

之前：

- 权限规则只是 `PermissionContext.rules` 上的一串列表，没有显式来源和优先级
- `TodoWrite` 只维护进程内的 todo 状态，不会落到结构化任务系统
- session 虽然能保存恢复，但 providerless/mock LLM、消息对象恢复、context usage 恢复还有兼容口子

现在：

- 权限规则有了显式 `PermissionStore`，可以按 `source + priority` 组织
- `TodoWrite` 可以直接同步到 `TaskService`，todo 只是结构化任务的兼容视图
- session persistence 全链路测试已打绿，消息对象、context usage、providerless LLM 恢复都能闭环

## 已落地的点

1. `PermissionStore`

- 新增 `core/permissions/store.py`
- 支持按 source 维护规则，并显式定义 source priority
- `PermissionContext` 现在会把 store 里的规则同步成有效规则视图

2. `accept_edits / dont_ask` 语义补齐

- `accept_edits` 会直接放行文件读写类工具
- `dont_ask` 会自动拒绝需要确认或高风险的工具

3. `TodoWrite -> TaskService`

- `TodoWriteTool` 支持绑定 `TaskService`
- 每次 todo 全量更新会同步结构化任务，并把“当前 todo 视图”映射到任务 metadata
- 被移出 todo 视图的任务不会被删除，而是标记为 `visible = false`

4. `session restore` 兼容层补齐

- `ConversationStore` 会把简单消息恢复成真实 `Message` 对象
- 自定义工具但未提供 registry 时，不再硬恢复成空 registry
- providerless/mock LLM 的 plain invoke 与会话恢复已兼容
- `context usage` 会缓存并随 session 恢复

## 一个具体过程例子

假设你有一个 code agent，要改 `workspace/target.py`。

旧行为里，模型通常会：

- 用 `TodoWrite` 维护一个内存 todo 列表
- 再单独用 `TaskCreate/TaskUpdate` 维护结构化任务
- session 恢复后，todo 和 task 可能对不上

现在可以变成：

1. 模型先调用 `TodoWrite`
2. `TodoWrite` 直接把列表同步进 `TaskService`
3. 结构化任务成为事实源，todo 只是一层兼容摘要视图
4. 如果此时 `save_session`，恢复后 `TaskList`、权限规则、context usage 都还是同一套状态

这意味着后面的 `Runtime Core` 和 `Collaboration Layer` 不需要再额外兼容两套任务来源。

## 真实 example

真实 example 已放在：

- [example_phase1a_permission_task_session.py](/home/wxd/LLM/EasyAgent/example/example_phase1a_permission_task_session.py)

这个 example 用的是你指定的真实 LLM：

```python
EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
```

它演示的是真实过程：

- 给 `FileWrite/FileEdit` 配置“默认要确认，但工作区内允许”的权限规则优先级
- 用 task-backed `TodoWrite` 管理当前修改任务
- 让 agent 修改一个真实 workspace 目录下的文件
- 保存 session 后再加载，检查任务、权限和 context usage 是否保持一致

我没有执行这个 example，保留给你后续自己调试。
