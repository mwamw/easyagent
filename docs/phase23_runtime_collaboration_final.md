# Phase 2/3 Final: Runtime Lifecycle + Collaboration Completion

本文档记录 `Phase 2: Runtime Core` 与 `Phase 3: Collaboration Layer` 在当前仓库中的收口状态。

这一轮完成的不是“再多几个 tool”，而是把多 agent runtime 真正补到可编排、可查询、可等待、可停止、可恢复的框架形态。

## 本轮完成了什么

### 1. Subagent 生命周期闭环

`runtime/agents/` 现在已经覆盖：

- 创建
- 查询
- 列表
- 等待
- 协作停止
- 删除 runtime handle
- session 恢复

对应实现包括：

- `AgentRuntimeManager.wait()`
- `AgentRuntimeManager.stop()`
- `AgentRuntimeManager.delete_handle()`
- `SubagentManager.wait()`
- `SubagentManager.stop()`
- `SubagentManager.delete()`

其中 `stop` 不是“直接改个状态”，而是协作停止协议：

- 如果 future 还没真正开始执行，直接取消并进入 `stopped`
- 如果子 agent 已经运行，则向 agent 发送 stop request
- 默认 `BasicAgent` 子 agent 会在后续执行边界检查 stop request，并以 `stopped` 终止

这意味着 stop 是真实协议，不是测试用假状态。

### 2. 公开 runtime tools 补齐

新增四个正式 tool：

- `AgentGet`
- `AgentList`
- `AgentWait`
- `AgentStop`

现在上层 code agent 可以直接把后台子 agent 当成一等 runtime 对象编排，而不是只能靠 `Agent` 启动以后自己猜状态。

### 3. BackgroundAgentHandle 正式建模

除了 `AgentHandle`，现在还补了 `BackgroundAgentHandle`：

- `isBackground`
- `canWait`
- `canStop`
- `stopRequested`

这样后台子 agent 的状态就不是“普通 handle + 约定俗成”，而是明确的 runtime 类型。

### 4. task-agent-team 绑定打通

这一步把任务系统和多 agent runtime 之间的关系做实了。

当父 agent 已绑定 `TaskService` 且调用 `Agent` 委派子任务时：

- 会自动创建一个 child task
- child task 的 `parent_task_id` 指向当前父任务
- child agent 的 `ExecutionContext.current_task_id` 指向这个 child task
- task metadata 会记录：
  - `agentId`
  - `teamId`
  - `teamName`
  - `outputFile`
  - `runtime status`

同时：

- `AgentGet / AgentList / AgentWait / AgentStop` 会在读取 handle 时继续同步 task 状态
- background agent 完成、停止、报错后，task 状态会跟着更新

这让 `task -> agent -> team` 不再是三套松散信息，而是能互相映射。

### 5. mailbox 支持 task 关联消息

`SendMessage` 现在除了：

- `recipient_type="agent"`
- `recipient_type="team"`

还支持：

- `recipient_type="task"`

这类消息会投递给所有 `ExecutionContext.current_task_id == recipient_id` 的子 agent。

所以 mailbox 现在已经支持：

- 点对点消息
- 团队广播
- 任务范围广播

### 6. session restore 与 collaboration runtime 一致恢复

上一轮已经能恢复 runtime/team/mailbox 基础状态；这一轮继续补齐后，恢复链路现在能覆盖：

- runtime handles
- background agent 标识
- team members
- mailbox
- child task 绑定
- `Agent / AgentGet / AgentList / AgentWait / AgentStop / SendMessage / TeamCreate / TeamDelete`

也就是说，会话恢复后不是只有“还能聊天”，而是协作 runtime 还在。

### 7. tool 返回协议补齐给 LLM 的可见性

这一轮还补了 `Phase 2/3` 相关 tool 的返回可见性问题。

此前这些 tool 虽然已经把完整 payload 放进了 `structured_data`，但 tool loop 回灌给 LLM 的其实是 `ToolResult.to_display_string()`。如果 tool 只返回一句简短 `content`，LLM 实际看不到完整 handle、team、message、task payload。

现在这批 tool 都改成了“摘要 + 结构化 payload”一起进入 display text：

- `Agent`
- `AgentGet`
- `AgentList`
- `AgentWait`
- `AgentStop`
- `SendMessage`
- `TeamCreate`
- `TeamDelete`
- `TaskCreate`
- `TaskGet`
- `TaskUpdate`

这样 LLM 在下一轮编排时，能直接看到：

- `outputFile`
- `executionContext`
- `teamId / teamName`
- `currentTaskId`
- `deliveries`
- `requestedReason / timedOut`
- 完整 task object

## 现阶段框架发生了什么变化

在这一步之前，EasyAgent 的多 agent 能力更像：

- 可以委派
- 可以广播
- 可以 worktree 隔离
- 但主 agent 对后台子 agent 的后续控制很弱
- task 和 team 与 runtime 的关系也没有真正收口

在这一步之后，框架形态变成了：

- subagent 是一等 runtime handle
- handle 可 query / list / wait / stop / restore
- background agent 有明确类型与控制面
- task、agent、team 三者能互相映射
- mailbox 支持 task-scope message
- session restore 可以恢复整套协作现场

这时 EasyAgent 才真正具备了 Claude Code 风格 code agent 所需的多 agent runtime 骨架。

## 一个具体例子

下面是当前阶段推荐的真实流程：

1. 主 agent 创建 root task：`Phase 2/3 final validation`
2. 主 agent 创建团队 `phase23-final`
3. 主 agent 用 `Agent` 启动一个后台子 agent
4. 框架自动为该子 agent 创建 child task，并把 child task ID 写入其 `ExecutionContext.current_task_id`
5. 主 agent 用 `AgentList` 查看当前 background handles
6. 主 agent 用 `SendMessage(recipient_type="team")` 给团队广播约束
7. 主 agent 再用 `SendMessage(recipient_type="task")` 给这个 child task 范围下的 agent 广播补充要求
8. 主 agent 用 `AgentWait` 等待子 agent 完成
9. 如果另一个子 agent 长时间运行，主 agent 可用 `AgentStop(wait=true)` 请求它进入 `stopped`
10. 主 agent 保存 session
11. 新进程中 `load_session()` 后，仍可查询：
    - 团队
    - handle
    - mailbox
    - child task 绑定

## 真实 example

对应 example：

- `example/example_phase23_runtime_collaboration_final.py`

这个 example 使用真实 LLM：

```python
llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
```

它不会自动执行，但会真实演示：

- root task 创建
- team 创建
- background subagent 启动
- `AgentGet / AgentList / AgentWait / AgentStop`
- `SendMessage` 的 team/task 两种广播
- `save_session / load_session` 后的 runtime 恢复检查

## 现在可以认为完成的阶段

按当前 `walkthrough.md` 的定义，可以认为下面两期已经完成到当前目标：

- `Phase 2: Runtime Core`
- `Phase 3: Collaboration Layer`

后续下一阶段就是：

- `Phase 4: Code Intelligence v1`

也就是 `codeintel/` 的 LSP 版本。
