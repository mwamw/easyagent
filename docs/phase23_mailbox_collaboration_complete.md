# Phase 2/3 Mailbox Collaboration Complete

本文档记录本轮完成的能力：把 `current_execution_plan.md` 中多智能体协作尚未闭合的部分补齐，重点是 mailbox 消费协议、子 agent 自动看到 mailbox 输入、后台完成记录，以及协作类复杂工具更详细的提示词。

## 本轮完成了什么

### 1. mailbox 从“只存消息”升级成“可消费协作输入”

新增：

- `MailboxRead`
- `MailboxAck`

message 生命周期现在是：

- `queued`
- `delivered`
- `consumed`
- `expired`

这意味着：

- `SendMessage` 不再只是“往 handle 上塞一条静态消息”
- runtime 现在能区分“已送达”和“已消费”
- 上层 manager 可以检查协作是否真正发生，而不是只检查消息是否发出去

对应实现：

- `runtime/agents/manager.py`
- `runtime/agents/models.py`
- `Tool/builtin/mailbox_tools.py`

### 2. 子 agent 在执行循环里会自动看到 mailbox 输入

`BaseAgent` 现在会在构建 system prompt 时自动读取自己的 pending mailbox，并注入一个 `## 协作邮箱` prompt block。

这意味着：

- team 广播和 task 广播不再只是 manager 侧状态
- 子 agent 后续每一轮请求都会把 mailbox 当成运行时输入的一部分
- 子 agent 读完后可以显式调用 `MailboxAck` 确认消费

对应实现：

- `core/agent.py`
- `agent/components/prompt_composer.py`
- `Tool/builtin/agent_tool.py`

### 3. background agent completion records 补齐

runtime 现在维护 `completion records`。

这意味着：

- 宿主不必强依赖 `AgentWait` 才知道某个后台 agent 已完成
- 可以直接轮询 `agent_runtime.list_completion_records()` 获取新完成的 background agent
- 这些记录也会进入 runtime state 导出

对应实现：

- `runtime/agents/models.py`
- `runtime/agents/manager.py`

### 4. 协作类复杂工具的提示词补详细

这轮重点补了下面这些工具的 schema prompt / guidance：

- `Agent`
- `AgentGet`
- `AgentList`
- `AgentWait`
- `AgentStop`
- `SendMessage`
- `TeamCreate`
- `TeamDelete`
- `MailboxRead`
- `MailboxAck`

目标不是“多写字”，而是让模型真正知道：

- 什么时候应该调用
- 什么时候不该调用
- 调用后的状态语义是什么
- 下一步该接什么工具

## 现阶段框架的变换

### 之前

EasyAgent 的多智能体协作更接近：

- 能启动子 agent
- 能把消息投递到 runtime mailbox
- 能保存 team / mailbox / execution_context

但子 agent 本身并不会自动消费这些消息，因此“协作”更多只是宿主侧的状态管理。

### 现在

EasyAgent 的多智能体协作已经变成：

- manager 可以启动 background agent
- manager 可以向 agent / team / task 发送运行时消息
- 子 agent 会在自己的执行循环里自动看到 mailbox 输入
- 子 agent 可以显式读取 mailbox 结构化载荷并确认消费
- 宿主可以通过 completion records 轮询后台完成状态

也就是说，协作信息已经真正进入 agent execution path，而不只是留在 runtime handle 上。

## 一个具体例子

下面是这轮能力对应的真实工作流：

1. manager 先用 `TeamCreate("review-team")` 建团队。
2. manager 用 `Agent(run_in_background=true, team_name="review-team")` 启动一个后台 worker。
3. worker 先读取第一个文件，进入下一轮推理。
4. manager 这时调用 `SendMessage(recipient_type="team", ...)`，追加一条“先总结结论，再补理由；并额外查看 example_stream.py”的消息。
5. worker 下一轮请求时，system prompt 中会自动出现 `## 协作邮箱`，看到这条新要求。
6. 如果 worker 需要完整结构化载荷，可以显式调用 `MailboxRead`。
7. worker 把新要求纳入执行后，用 `MailboxAck` 把对应消息标成 `consumed`。
8. manager 可以继续用 `AgentWait` 等完成，或者用 `agent_runtime.list_completion_records()` 轮询已完成的后台 worker。

这个流程说明了本轮改造的核心：`SendMessage` 已经不再只是“写一条消息”，而是能真实改变子 agent 后续行为。

## 真实 example

本轮新增的手动调试 example：

- `example/example_phase23_mailbox_collaboration_complete.py`

这个 example 会真实使用：

- `EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="122", model="qwen3.5-9b")`
- `Agent`
- `AgentGet / AgentList / AgentWait / AgentStop`
- `SendMessage`
- `MailboxRead / MailboxAck`
- `TeamCreate / TeamDelete`

它不会被自动执行，供后续手动调试。

## 验证

本轮补了两类契约测试：

- runtime / collaboration：
  - `test_runtime_mailbox_read_ack_and_completion_records`
  - `test_mailbox_tools_and_prompt_injection_complete_message_lifecycle`
- session restore：
  - `test_basic_agent_session_restore_rebuilds_collaboration_runtime`

这些测试覆盖：

- send -> read -> ack
- mailbox prompt 自动注入
- completion records
- collaboration runtime restore 后 mailbox tools 自动恢复

## 结论

按 `current_execution_plan.md` 的口径，本轮完成后：

- `Phase A：协作闭环补齐` 已完成
- `Phase B：Mailbox 消费与 Team 协作语义补齐` 已完成

下一阶段应进入：

- `Phase C：Runtime Lifecycle 与 Restore Report`
