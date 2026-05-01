# Runtime Collaboration Guide

Runtime 模块负责“一个 Agent 如何管理其他 Agent，以及这些 Agent 如何共享上下文、任务、团队和消息”。  
如果你只是做单智能体问答，它不是第一优先级；如果你要做 Code Agent、后台审计 worker、team broadcast、长任务管理，它就是核心模块。

相关文档：

- [Agent Guide](./agent_guide.md)
- [Tasks Guide](./tasks_guide.md)
- [Permissions Guide](./permissions_guide.md)
- [Worktree Guide](./worktree_guide.md)

## 1. Runtime 解决什么问题

如果没有 runtime，子 agent 往往只是“一次性的函数调用包装”：

- 没有统一 ID
- 没有可查询状态
- 没有 mailbox
- 没有 background 生命周期
- 没有 team 组织
- 没有恢复语义

EasyAgent 的 runtime 目标是把这些能力统一起来。

## 2. 核心对象

### `AgentRuntimeManager`

这是多 Agent 运行时的入口，负责：

- 启动 foreground / background 子 agent
- 保存 `SubagentRequest`
- 维护 mailbox
- 维护 completion record
- 把 agent 挂到 team 上
- 恢复 runtime 快照

### `ExecutionContext`

描述“这次 agent 在什么上下文里运行”。主要字段包括：

- `workspace_root`
- `allowed_roots`
- `mcp_servers`
- `execution_mode`
- `permission_mode`
- `current_task_id`
- `worktree_path`
- `worktree_branch`
- `metadata`

这不是 prompt 文本，而是 runtime 层的结构化环境信息。

### `AgentHandle`

表示一个 agent 当前的运行状态快照，常见字段有：

- `agent_id`
- `status`
- `description`
- `prompt`
- `output_file`
- `workspace_root`
- `allowed_roots`
- `execution_context`
- `team_name / team_id`
- `started_at / finished_at`
- `error / stop_reason`
- `usage`
- `mailbox`
- `metadata`

如果是后台 agent，则通常使用：

### `BackgroundAgentHandle`

它在 `AgentHandle` 基础上额外暴露：

- `is_background`
- `can_wait`
- `can_stop`
- `stop_requested`

### `MailboxMessage`

结构化消息，不只是字符串。关键字段有：

- `message_id`
- `sender_id`
- `recipient_type`
- `recipient_id`
- `content`
- `status`
- `delivered_at`
- `consumed_at`
- `expires_at`
- `acked_by`
- `metadata`

### `CompletionRecord`

记录已完成 agent 的摘要结果，适合：

- 列表页
- 状态追踪
- 恢复后的快速回看

### `TeamManager`

负责团队的生命周期：

- `create_team`
- `get_team`
- `list_teams`
- `add_member`
- `remove_member`
- `delete_team`

### `TeamHandle`

团队的结构化快照，至少包含：

- `team_id`
- `name`
- `description`
- `member_agent_ids`
- `created_at`
- `metadata`

## 3. 子 Agent 启动后的完整流程

一个典型的子 agent 生命周期通常是：

1. 主 agent 通过 `Agent` 工具提交 `SubagentRequest`
2. `AgentRuntimeManager` 生成 `agent_id`
3. 组装 `ExecutionContext`
4. 如有 team，绑定 `team_id / team_name`
5. 如配置为 background，返回 `BackgroundAgentHandle`
6. 子 agent 运行期间状态可通过 `AgentGet / AgentList / AgentWait` 查询
7. 如有 mailbox，新消息会挂到 handle 上
8. 结束后生成 `CompletionRecord`

这和简单的“spawn 一个线程然后等结果”不同，运行时是有结构化状态的。

## 4. Mailbox 的作用

Mailbox 的意义是让 agent 间沟通不必再硬编码到初始 prompt 里。

适合场景：

- manager 先下发任务，再追加补充约束
- team 广播统一格式要求
- worker 完成后 ack 某条消息
- 外部系统在 agent 运行中途动态注入上下文

常见操作：

- `MailboxRead`
- `MailboxAck`
- `SendMessage`

推荐模式：

- manager 通过 `SendMessage` 广播规则
- worker 在关键节点主动读 mailbox
- 使用 ack 标识已经消费的消息

## 5. Team 的作用

Team 不是必须的，但在多 agent 产品里很有用。  
它解决的是“谁属于同一组协作者”。

适合：

- reviewer team
- research team
- runtime / testing / docs 分工团队

典型流程：

1. `TeamCreate`
2. `Agent` 启动子 agent 时指定 `team_name`
3. `SendMessage` 对 team 广播
4. team 里的 agent 按需读取 mailbox

## 6. 和 Task 的关系

runtime 与 task 系统通常一起出现：

- `ExecutionContext.current_task_id` 记录当前任务
- 子 agent 可自动挂到父任务或子任务
- completion record 也可以带 `current_task_id`

所以推荐你把：

- runtime
- task
- mailbox

一起设计，而不是分开想。

详见：

- [Tasks Guide](./tasks_guide.md)

## 7. 和 Permission / Worktree 的关系

### Permission

子 agent 不应该默认“权限全开”。  
更合理的方式是：

- 父 agent 传入 `permission_context`
- runtime 在启动子 agent 时继承或覆盖权限模式

### Worktree

如果你要让多个子 agent 并行修改代码，worktree 通常是必选项。  
否则不同 agent 可能直接写进同一工作树。

详见：

- [Worktree Guide](./worktree_guide.md)

## 8. 如何接到 `BasicAgent`

最常见的接法是：

```python
agent = BasicAgent(
    name="manager",
    llm=llm,
    enable_tool=True,
    tool_registry=registry,
    agent_runtime=agent_runtime,
    team_manager=team_manager,
    task_service=task_service,
)
```

然后给 registry 注册这些工具：

- `register_agent_tool`
- `register_agent_runtime_tools`
- `register_send_message_tool`
- `register_mailbox_tools`
- `register_team_create_tool`
- `register_team_delete_tool`

## 9. 一个推荐的产品装配

如果你做的是多 Agent Code Agent，推荐 runtime 组合如下：

1. `AgentRuntimeManager`
2. `TeamManager`
3. `TaskService`
4. `WorktreeManager`
5. `PermissionContext`
6. `SessionStore`

这样你就同时有：

- 协作
- 可中断
- 可恢复
- 隔离执行
- 结构化任务推进

## 10. 常见坑

### 把 runtime 当成“工具集合”

不对。  
runtime 是状态层，工具只是它的控制面。

### 只有 `Agent` 工具，没有 `AgentGet / Wait / Stop`

这会让后台 agent 失去可观测性和可控性。

### 忽略 mailbox

如果所有补充要求都只能写死在初始 prompt，中途调度能力会很差。

### 子 agent 没继承 execution context

会导致：

- workspace_root 不一致
- allowed_roots 不一致
- current_task_id 丢失
- permission_mode 不一致
