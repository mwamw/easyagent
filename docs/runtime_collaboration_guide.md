# Runtime Collaboration Guide

`MultiAgentRuntime` 是可选多智能体模块。它把 subagent、background lifecycle、team、mailbox、task 绑定、session 恢复和控制工具放在一个 façade 下。

## 1. 安装

```python
agent.with_multi_agent(
    workspace_root=".",
    storage_dir=".easyagent/agents",
    max_background_tasks=4,
)
```

如果 Agent 尚未安装 ToolRegistry，该操作会自动安装，因为多智能体控制面由工具暴露给模型。

也可以传入自定义实现：

```python
agent.with_multi_agent(MyMultiAgentRuntime())
```

自定义实现必须继承 `BaseMultiAgentRuntime`。

## 2. 运行时对象

### ExecutionContext

统一描述：

- workspace/allowed roots
- execution/permission mode
- current task
- worktree path/branch
- 可见 MCP servers
- runtime metadata

父子 Agent 和 worktree 使用同一类上下文，不再分别传递松散参数。

### AgentHandle

查询结果包含 agent ID、状态、prompt、output file、execution context、team/task、usage、mailbox、error 和 stop reason。

### BackgroundAgentHandle

额外描述 `is_background`、`can_wait`、`can_stop` 和 `stop_requested`。

### TeamHandle

保存 team ID、名称、描述、成员和 metadata。

### MailboxMessage

保存 sender、recipient、content、task metadata、delivered/consumed/expired 状态和时间。

## 3. 工具控制面

安装模块后自动注册：

| 工具 | 用途 | 关键返回 |
| --- | --- | --- |
| `Agent` | 启动前台或后台子 Agent | `agentId/status/outputFile/executionContext` |
| `AgentGet` | 查询单个 Agent | 完整 handle |
| `AgentList` | 过滤并列出 Agent | `count/agents` |
| `AgentWait` | 等待后台 Agent | handle、`timedOut` |
| `AgentStop` | 请求停止 Agent | status、stop reason |
| `TeamCreate` | 创建 team | team handle |
| `TeamDelete` | 删除 team | team 和原成员列表 |
| `SendMessage` | 发给 agent/team/task | delivery count 和 deliveries |
| `MailboxRead` | 查看 mailbox | message records |
| `MailboxAck` | 标记消息已消费 | ack records |

复杂工具的 description 包含何时调用、参数含义、前后台差异、返回字段和后续动作。模型可以从 `outputFile` 读取后台完整输出，而不是只得到“已启动”。

## 4. 子 Agent 生命周期

1. `Agent` 创建 `SubagentRequest`。
2. Runtime 生成 ID 和 output file。
3. 从父 Agent 复制 workspace、roots、mode、permission、task、MCP 和 worktree context。
4. 创建子 Agent 并记录 handle。
5. 前台调用直接完成；后台调用立即返回可查询 handle。
6. `AgentGet/List/Wait/Stop` 管理后续状态。
7. 结束时写 output file 和 completion record。
8. session snapshot 保存 handle、team、mailbox 和 completion state。

## 5. Mailbox 如何被子 Agent 读取

`SendMessage` 只负责投递。真正进入模型上下文的链路是：

1. executor 在每个 LLM request 前调用 `MultiAgentRuntime.sync_mailbox()`。
2. runtime 按当前 `ExecutionContext.metadata.agentId` 读取未消费消息。
3. 消息状态更新为 delivered。
4. 每条消息转换成带唯一 dedup key 的永久 MetaMessage。
5. MetaMessage manager 在安全 checkpoint 将其追加到 canonical history 尾部。
6. 当前 request 的 replay 因此立即包含消息。
7. 后续同步不会重复插入相同 message ID。

`MailboxRead` 和 `MailboxAck` 仍提供结构化控制：前者让模型检查状态，后者将消息标记为 consumed。Ack 不删除已经进入历史的消息，因为它已经成为协作过程的一部分。

## 6. Team 和 Task

Team 用于编组和广播；Task 用于工作分解和 owner 关系。AgentTool 启动子 Agent 时可以：

- 指定 team
- 基于父 task 创建 child task
- 将 child task owner 绑定到子 Agent ID
- 把 team/task 信息写回 handle 和 Task metadata

要启用结构化任务：

```python
from easyagent.tasks import InMemoryTaskStore, TaskService

agent.with_task_service(TaskService(InMemoryTaskStore()))
agent.with_multi_agent(workspace_root=".")
```

## 7. Worktree

父 Agent 安装 Worktree 后，ExecutionContext 保存 worktree path/branch。子 Agent 从 context 继承隔离工作区。并行修改代码时应显式设计一个 Agent 对应一个 worktree，避免多个 worker 写同一目录。

## 8. 恢复和关闭

`MultiAgentRuntime.export_state()` 同时导出 AgentRuntime 和 TeamManager 状态。恢复报告会聚合 restored、degraded、missing items 和 issues，不会静默吞掉某个子组件失败。

Agent 关闭时先关闭 MultiAgent，停止或回收后台任务，再关闭 MCP、CodeIntel、Worktree、Observability 和 LLM。

## 9. 上层直接控制

产品侧需要绕过模型工具时，可以访问：

```python
runtime = agent.multi_agent.agent_runtime
handle = runtime.get_handle(agent_id)
handle = runtime.wait(agent_id, timeout_ms=30000)
deliveries = runtime.send_message(
    recipient_type="agent",
    recipient_id=agent_id,
    content="补充约束",
    sender_id=agent.name,
)
```

模型侧和产品侧操作同一个 runtime 事实源。
