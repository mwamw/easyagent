# Phase C Restore Report And Lifecycle

本文档记录本轮完成的能力：把 EasyAgent 的 session/runtime 恢复从“能把结构读回来”升级成“能明确表达恢复边界、降级状态和 worktree/runtime 生命周期结果”。

## 本轮完成了什么

### 1. `SessionRestoreReport` 正式落地

新增：

- `core/session/report.py`
- `core/session/__init__.py`

现在 `load_session()` 恢复后，agent 会携带一份结构化 restore report，可通过：

- `agent.last_restore_report`
- `agent.get_last_restore_report()`

读取。

报告会覆盖：

- 总体恢复状态：`restored / degraded / failed`
- `execution_context` 是否恢复成功
- 缺失工具
- 缺失 skill
- runtime / team / worktree 等组件级恢复结果

### 2. runtime restore 不再静默降级

`AgentRuntimeManager.restore_state()` 和 `SubagentManager.restore_state()` 现在都会返回结构化报告。

这意味着：

- 如果某个 background subagent 在保存会话时还没完成，恢复后会明确标成 `interrupted`
- runtime restore report 会明确指出：
  - 哪些 handle 完整恢复
  - 哪些 background agent 是降级恢复
  - 哪些 mailbox / team assignment 成了 orphan
  - 当前 completion records 数量

对应实现：

- `runtime/agents/manager.py`
- `Tool/runtime/subagent_manager.py`

### 3. worktree 也进入正式 restore 协议

`WorktreeManager` 现在支持：

- `export_state()`
- `restore_state()`
- `close()`

同时会跟踪：

- 已管理的 worktree
- 当前 active session
- active session 是否还能恢复
- 缺失 worktree 是否需要降级报告

这意味着：

- session snapshot 不再只能记住 `execution_context.worktreePath`
- worktree 恢复时能知道“这个路径还存在”还是“路径没了，只能降级”

对应实现：

- `Tool/runtime/worktree_manager.py`

### 4. close 语义正式收口

本轮补了 runtime 侧的 close 报告：

- `SubagentManager.close()` 会报告关闭时是否仍有未终止后台子 agent
- `AgentRuntimeManager.close()` 会报告 unresolved background handles，并保留 subagent close report
- `WorktreeManager.close()` 会基于 `keep/remove` 策略收口 active worktree session

这让“关闭运行时”也不再是黑盒动作。

### 5. `BaseAgent.close()` 正式落地

此前 close 语义只存在于 manager 层。现在 `BaseAgent` 也有统一 cleanup 入口：

- `agent.close()`
- `agent.get_last_close_report()`

它会把：

- `agent_runtime.close()`
- `worktree_manager.close()`
- `llm.close()`

统一汇总为一份结构化 close report。

这意味着上层产品不需要自己知道 runtime / worktree / provider 的细节，就能拿到：

- 是否完整关闭
- 是否存在 degraded close
- 哪个组件关闭失败
- worktree 是 `keep` 还是 `remove`

## 现阶段框架的变换

### 之前

以前的 `load_session()` 更像：

- 能把 session snapshot 读出来
- 能把 runtime/team/mailbox 结构重新绑定回 agent
- 但恢复过程中如果有降级，基本靠日志 warning 猜

对于上层产品来说，这有两个明显问题：

1. 看不到哪些 background agent 只是“结构恢复”，其实已经不能续跑  
2. 看不到 worktree/runtime 恢复到底是完整还是降级

### 现在

现在 `load_session()` 会给出一份正式 restore report。

这意味着上层产品可以直接基于它做：

- UI 提示
- resume 决策
- degraded runtime 警告
- 是否要求用户重新启动后台任务
- 是否提示 worktree 已丢失
- close 阶段的资源回收提示

换句话说，EasyAgent 现在不只是“支持恢复”，而是“支持解释恢复结果”。

## 一个具体例子

下面是这轮能力对应的真实工作流：

1. manager 在一个 git repo 工作区里创建并进入 worktree。
2. manager 启动一个 background subagent。
3. 这个 background subagent 还没完成时，manager 调用 `save_session()`。
4. 新进程里执行 `load_session()`。
5. 恢复后的 agent 读取 `get_last_restore_report()`。
6. 报告会明确告诉你：
   - `agent_runtime` 是不是 degraded
   - 哪个 background agent 被恢复成了 `interrupted`
   - `worktree_runtime` 是 restored 还是 degraded
   - 有没有缺失工具、缺失 skill
7. 当这一轮调试结束后，再调用 `agent.close()`，拿到 cleanup report，确认 runtime/worktree/llm 是否都已正常收口。

这个流程的意义是：恢复之后你不再需要靠猜测来判断“这个 session 还能不能续跑”。

## 真实 example

本轮新增的手动调试 example：

- `example/example_phasec_restore_report_lifecycle.py`

它会真实使用：

- `EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="122", model="qwen3.5-9b")`
- `Agent`
- `AgentGet / AgentList / AgentWait / AgentStop`
- `EnterWorktree / ExitWorktree`
- `save_session / load_session`
- `get_last_restore_report()`
- `close() / get_last_close_report()`

它不会被自动执行，供后续手动调试。

## 验证

本轮新增或补强的契约测试覆盖了：

- `WorktreeManager` 的 export/restore/close 报告
- background runtime 的 degraded restore 报告
- collaboration runtime restore report
- worktree runtime restore report

关键测试包括：

- `test_worktree_manager_restore_report_round_trip`
- `test_worktree_manager_restore_report_marks_missing_worktree`
- `test_basic_agent_session_restore_reports_degraded_background_runtime`
- `test_basic_agent_session_restore_reports_worktree_runtime`
- `test_basic_agent_session_restore_rebuilds_collaboration_runtime`
- `test_basic_agent_close_returns_cleanup_report`
- `test_basic_agent_close_reports_degraded_background_runtime`

## 结论

按 `current_execution_plan.md` 的口径，本轮完成后：

- `Phase C：Runtime Lifecycle 与 Restore Report` 的核心目标已完成

下一阶段应进入：

- `Phase D：Code Intelligence v1`
