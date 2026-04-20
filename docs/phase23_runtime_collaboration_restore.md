# Phase 2/3: Runtime / Collaboration Session Restore

> 这是中间阶段文档，记录的是 session restore 接线完成时的状态。`Phase 2 / Phase 3` 最终版请看 `docs/phase23_runtime_collaboration_final.md`。

本文档记录本轮完成的能力：把 `runtime + team + mailbox + execution_context` 正式接入 `save_session/load_session`，让多 agent 协作状态不再只是进程内临时对象。

## 这一步完成了什么

- `ExecutionContext` 新增了 `from_dict()`，可以从 session snapshot 直接恢复。
- `SubagentRequest`、`SubagentSnapshot`、`MailboxMessage`、`TeamHandle` 都补了序列化/反序列化能力。
- `SubagentManager` 新增：
  - `export_state()`
  - `restore_state()`
- `AgentRuntimeManager` 新增：
  - `export_state()`
  - `restore_state()`
- `TeamManager` 新增：
  - `export_state()`
  - `restore_state()`
- `BaseAgent.save_session()` 现在会把以下信息一起保存进 snapshot：
  - `execution_context`
  - `collaboration_runtime.agent_runtime`
  - `collaboration_runtime.teams`
- `BaseAgent.load_session()` 现在会自动重建并绑定：
  - `agent_runtime`
  - `team_manager`
  - `execution_context`
- 如果会话原本启用了 `Agent / SendMessage / TeamCreate / TeamDelete`，恢复后这些工具也会自动重新注册到当前 agent 的 `ToolRegistry`。

## 现阶段框架发生了什么变化

在这一步之前，EasyAgent 的多 agent 协作主要是“当前进程里能跑起来”：

- 可以启动子 agent
- 可以创建 team
- 可以发 mailbox 消息
- 但一旦 session 保存再恢复，这些运行时关系基本就丢了

在这一步之后，框架形态变成了“协作运行时是可持久化的一等状态”：

- 子 agent handle 不再只是工具执行结果，而是 session 可恢复的 runtime 记录
- team 成员关系可以跨 session 保留
- mailbox 消息可以跨 session 保留
- `current_task_id`、`worktree`、`permission/mode` 对应的 `execution_context` 也能随 handle 一起恢复
- 上层 code agent 可以真正依赖这套 runtime 做“中断后继续查看状态”，而不是每次从零建协作上下文

## 一个具体过程例子

这一步推荐的真实流程如下：

1. 主 agent 创建结构化任务，并把自己切到该任务上下文。
2. 主 agent 创建团队 `restore-reviewers`。
3. 主 agent 调用 `Agent` 启动一个子 agent，并设置 `team_name="restore-reviewers"`。
4. 子 agent 得到继承后的 `ExecutionContext`：
   - `workspaceRoot`
   - `allowedRoots`
   - `executionMode`
   - `permissionMode`
   - `currentTaskId`
5. 主 agent 用 `SendMessage` 向整个团队广播一条消息。
6. 主 agent 调用 `save_session()`。
7. 新进程里调用 `BasicAgent.load_session()` 恢复。
8. 恢复后的 agent 可以直接查询：
   - 团队是否还存在
   - 该子 agent 是否仍在 runtime 记录中
   - mailbox 消息是否还在
   - 该 handle 是否仍绑定原任务 ID

这不是“为了测试恢复一个 dict”，而是为了让后续 code agent 真能有长期协作态。

## 真实 example

对应的真实 example 文件：

- `example/example_phase23_runtime_collaboration_restore.py`

这个例子使用真实的 `EasyLLM(...)` 配置，流程包含：

- 创建主 agent
- 绑定任务系统
- 注册 `Agent / SendMessage / TeamCreate / TeamDelete`
- 让主 agent 真实创建 team、委派子 agent、发送团队消息
- 保存 session
- 再用 `BasicAgent.load_session()` 恢复
- 打印恢复后的 team、handle、mailbox、task 绑定情况

该文件不会在仓库里自动执行，后续可以手动调试。

## 这一轮之后还没完成的内容

这一步解决的是“协作运行时可恢复”，还不是 `Phase 2/3` 的最终版。剩余缺口主要还有：

- 更完整的后台子 agent 停止/恢复协议
- live background execution 的真正续跑，而不是 session 级状态恢复
- task / agent / team 的更强查询接口
- 更细的 restore report 和 lifecycle hardening
- `codeintel/` 的 LSP 版本

所以这一步的意义是：把协作 runtime 从“临时能力”变成“可恢复内核状态”，为下一步继续做 runtime hardening 和 code intelligence 打地基。
