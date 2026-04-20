# Phase 2 Runtime Core

> 这是中间阶段文档。`Phase 2 / Phase 3` 的最终收口结果请看 `docs/phase23_runtime_collaboration_final.md`。

本文档记录 EasyAgent 在 Phase 2 当前阶段完成的运行时协作能力，以及这一步对框架形态带来的变化。

## 这一步完成了什么

- 新增统一 `runtime/` 包，提供：
  - `ExecutionContext`
  - `AgentRuntimeManager`
  - `AgentHandle`
  - `MailboxMessage`
  - `TeamManager`
  - `TeamHandle`
- `AgentTool` 不再只直接操作 `SubagentManager`，而是升级为通过 `AgentRuntimeManager` 启动和查询子 agent。
- 新增 `SendMessage`、`TeamCreate`、`TeamDelete` 三个协作工具。
- 子 agent 现在会携带继承后的 `ExecutionContext`，包括：
  - `workspaceRoot`
  - `allowedRoots`
  - `executionMode`
  - `permissionMode`
  - `currentTaskId`
  - `worktreePath/worktreeBranch`

## 这一步带来的框架变换

在这一步之前，EasyAgent 的子 agent 更像“工具层临时任务”：

- 能启动
- 能后台跑
- 能用 worktree 隔离
- 但没有统一的 runtime handle
- 没有 mailbox
- 没有显式 team 概念
- 没有共享的 execution context 抽象

在这一步之后，EasyAgent 更接近“运行时框架”：

- 子 agent 通过 `AgentHandle` 暴露结构化状态
- agent 与 team 可以通过 mailbox 传递结构化消息
- `TeamManager` 可以持有一组 agent 成员
- `ExecutionContext` 统一表达当前工作区、权限模式、执行模式与任务上下文
- `AgentTool` 返回的不再只是结果文本，而是 runtime 级 payload

## 一个真实过程示例

下面这个过程就是当前阶段鼓励的真实协作流程：

1. 主 agent 调用 `TeamCreate(name="phase2-core")`
2. 主 agent 调用两次 `Agent`，分别启动：
   - `runtime-auditor`
   - `tests-auditor`
   并都设置 `team_name="phase2-core"`
3. 子 agent 启动后会被加入该 team，并各自得到结构化 `ExecutionContext`
4. 主 agent 调用 `SendMessage(recipient_type="team", recipient_id="<teamId>", ...)`
5. 该消息会投递到 team 成员的 mailbox
6. 主 agent 再根据各子 agent 的 `outputFile` 和 `agentId` 汇总结果

这个过程对应的可调试示例文件是：

- `example/example_phase2_runtime_team.py`

## 当前阶段还没完成的内容

这一步仍然不是 Phase 2 的最终版，还缺：

- `codeintel/` 的 LSP 版本
- 更完整的 subagent 停止/恢复协议
- team 与 session restore 的深度集成
- worktree/runtime/session 三者的一致恢复
- 后续 hooks、guardrails、restore report 等 hardening 能力
