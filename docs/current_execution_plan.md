# EasyAgent 当前执行计划

## 目标

这份计划基于当前仓库代码状态，而不是基于 `docs/walkthrough.md` 里的初始路线图重新从零排期。

框架定位保持不变：

- `code agent` 是一等公民
- 同时保留构建通用 agent 的抽象能力
- 优先补运行时闭环、恢复闭环、扩展协议，再补能力层
- 不优先做 CLI/TUI/voice 这类产品层能力

更新说明（2026-04-20）：

- `Phase A` 已完成：`AgentGet / AgentList / AgentWait / AgentStop`、后台 handle 语义、`completion records` 已补齐
- `Phase B` 已完成：`MailboxRead / MailboxAck`、message 生命周期、mailbox 自动注入 prompt、协作消费闭环已补齐
- `Phase C` 已完成：`SessionRestoreReport`、runtime/worktree restore report、`BaseAgent.close()` 生命周期收口已补齐
- 当前下一阶段应进入 `Phase D：Code Intelligence v1`

## 当前状态对照

### 已完成的基础能力

下面这些已经不是“待设计”，而是当前代码里已有实现：

- 权限系统已落地：`core/permissions/`
- plan/execute 模式机已落地：`core/execution_mode.py`
- 结构化任务系统已落地：`task/`
- Tool 执行链已接入权限引擎：`Tool/ToolRegistry.py`
- 会话状态已能保存 `mode / permission / current_task_id / runtime snapshot`：`core/agent.py`
- session 自动恢复框架内建依赖已落地：`core/agent.py`
- 多 agent runtime 基础对象已落地：
  - `runtime/context.py`
  - `runtime/agents/manager.py`
  - `runtime/agents/models.py`
  - `runtime/teams/manager.py`
- 协作工具已落地：
  - `Agent`
  - `AgentGet`
  - `AgentList`
  - `AgentWait`
  - `AgentStop`
  - `SendMessage`
  - `MailboxRead`
  - `MailboxAck`
  - `TeamCreate`
  - `TeamDelete`
- 多 agent 协作已具备 mailbox 消费闭环：
  - message 生命周期：`queued / delivered / consumed / expired`
  - 子 agent 可在执行循环中自动看到 mailbox 输入
  - runtime 支持 `completion records` 供宿主轮询后台完成事件
- provider schema adapter 已落地：`core/providers/tool_schema.py`

### 已完成但只到 MVP 的能力

这些模块已经存在，但还没有形成最终闭环：

- `AgentTool` 能启动同步/后台子 agent，也能挂 team/worktree
- `AgentRuntimeManager` 能保存 handle、mailbox、team assignment，并支持 export/restore
- `SubagentManager` 能同步执行、后台执行、保存 output file
- runtime/team 状态可以进入 session restore
- background runtime 恢复边界还没有结构化 restore report

### 当前最明显的缺口

这些缺口会直接限制 EasyAgent 继续长成 Claude Code 风格 code agent：

1. background runtime 生命周期还不完整

- 运行状态还偏轻量，缺少真正的一等公民 `BackgroundAgentHandle`
- 缺少后台任务完成通知、阻塞等待、停止、恢复后的降级报告
- session restore 能恢复 runtime 结构，但不能严谨表达“哪些后台执行无法真正续跑”

2. code intelligence 还没开始

- 没有 `codeintel/`
- 没有 LSP / symbol / diagnostics / workspace index
- code agent 仍主要依赖 `FileRead / Grep / Glob`

3. hooks / guardrails 还没开始

- 没有 `core/hooks/`
- 没有 `core/guardrails/`
- 当前只有 callbacks 和 permission rules，缺少内容级拦截与改写层

4. Tool 协议还没到最终版

- 已有 `risk_categories` 和 provider schema adapter
- 但还没有 `side_effect_level / resource_scope / visibility_scope / conflict policy`
- `ephemeral_context` 也还没有完整纳入 compaction / restore 协议

5. MCP 仍是轻量接入，不是 first-class runtime surface

- 现有 `mcp/runtime.py`、`mcp_client.py` 可以用
- 但没有 `connection_manager / auth / cache / policy`
- 也没有和权限系统、session/runtime 生命周期深度打通

6. SDK/package 边界还没收口

- 还没有 `pyproject.toml`
- 公共 API 边界还没有正式冻结
- `docs/` 和 `example/` 还没有按“框架示例 / 产品示例”重新整理

## 新的优先级原则

`walkthrough.md` 里的旧顺序是合理的长期路线，但按当前代码状态，优先级需要调整成：

1. 先做 `runtime lifecycle + restore report`
2. 再做 `codeintel`
3. 再做 `hooks/guardrails + tool protocol v2`
4. 再做 `MCP engineering`
5. 最后做 `SDK/package/doc` 收口

原因很简单：

- 现在多 agent 协作主闭环已经能跑通，最大的真实缺口变成“runtime 恢复边界和长期运行语义还不够清晰”
- 如果不先补 restore report、degraded runtime 表达和 lifecycle cleanup，后续 codeintel 和更复杂协作都很难长期稳定运行

## 新执行计划

### Phase A：协作闭环补齐

目标：把当前 Phase 2/3 从“能启动和发消息”补到“能协作完成任务”。

状态：已完成

#### 主要工作

- 新增 agent runtime 查询与控制工具：
  - `AgentGet`
  - `AgentList`
  - `AgentWait`
  - `AgentStop`
- 明确 `Agent` 在后台模式下的语义：
  - 启动即返回 `async_launched`
  - 后续必须通过 `AgentGet/Wait` 继续跟踪
- 为 runtime 增加显式状态集：
  - `async_launched`
  - `running`
  - `waiting`
  - `completed`
  - `error`
  - `stopped`
- 给 `AgentHandle` 增加更清晰的后台字段：
  - `is_background`
  - `completion_state`
  - `last_error`
  - `output_file`
  - `mailbox_count`
- 加入后台完成通知机制：
  - runtime 内部 event queue 或 completion records
  - 上层 agent/宿主能够知道哪个 background agent 已完成
- 让主 agent 能真正等待子 agent 完成后再汇总，而不是只汇总 handle

#### 建议落点

- `runtime/agents/models.py`
- `runtime/agents/manager.py`
- `Tool/builtin/agent_get.py`
- `Tool/builtin/agent_list.py`
- `Tool/builtin/agent_wait.py`
- `Tool/builtin/agent_stop.py`
- `Tool/builtin/__init__.py`

#### 验收标准

- 主 agent 可以启动两个后台子 agent，再用 `AgentWait` 等它们完成后汇总
- background subagent 的状态变化是可查询、可测试的
- `example_phase2_runtime_team.py` 能跑出“已完成的子 agent 输出”，不是只有 `agentId/outputFile`

### Phase B：Mailbox 消费与 Team 协作语义补齐

目标：让 mailbox 从“存消息”升级成“协作协议的一部分”。

状态：已完成

#### 主要工作

- 增加 mailbox 读取与消费能力：
  - `MailboxRead`
  - `MailboxAck` 或等价消费协议
- 定义 message 生命周期：
  - `queued`
  - `delivered`
  - `consumed`
  - `expired`
- 子 agent 在执行循环里能看到 mailbox 输入
- team broadcast 后，成员能读取消息并改变后续行为
- 支持 task 关联消息：
  - 消息绑定 `task_id`
  - team / agent / task 三者关系可查询
- 区分“消息送达成功”和“消息已被消费”

#### 建议落点

- `runtime/agents/models.py`
- `runtime/agents/manager.py`
- `Tool/builtin/send_message.py`
- `Tool/builtin/mailbox_read.py`
- `Tool/builtin/mailbox_ack.py`
- `core/agent.py` 或 tool loop 中的 mailbox 注入点

#### 验收标准

- team 广播后，至少一个子 agent 能读取到该消息
- mailbox 不再只是 handle 上的静态字段
- 协作测试覆盖“发送 -> 读取 -> 消费 -> 状态变化”全链路

### Phase C：Runtime Lifecycle 与 Restore Report

目标：把 runtime 从“能保存结构”补到“能表达恢复边界”。

状态：已完成，阶段说明见 `docs/phasec_restore_report_lifecycle.md`

#### 主要工作

- 明确 runtime/session 恢复语义：
  - 哪些子 agent 是已完成可恢复
  - 哪些子 agent 只是状态恢复，不能继续运行
  - 哪些后台线程已丢失，只能标记为 degraded
- 引入 `SessionRestoreReport`
- 引入 runtime restore report：
  - 恢复了哪些 handle
  - 哪些 background agent 不能真正续跑
  - 哪些 worktree/mcp/runtime mount 丢失
- 给 worktree runtime 补生命周期收口：
  - 创建后登记
  - agent stop/cleanup 时回收策略
- 补 agent runtime 的 stop / cleanup / close 语义
- 新增 `BaseAgent.close()`，把 runtime/worktree/llm cleanup 提升到 agent 层
- 把中断请求、confirmation、runtime waiting state 一起纳入 session 恢复边界

#### 建议落点

- `core/session/` 或 `core/agent.py` 内的 restore reporting 抽象
- `runtime/agents/manager.py`
- `runtime/teams/manager.py`
- `Tool/runtime/subagent_manager.py`
- `Tool/runtime/worktree_manager.py`

#### 验收标准

- 恢复后能明确知道 runtime 是“完整恢复”还是“降级恢复”
- 后台子 agent、mailbox、team assignment、execution context 的恢复结果有结构化报告
- session restore 不再静默跳过 runtime 丢失问题

### Phase D：Code Intelligence v1

目标：让 EasyAgent 具备真正可用的 code agent 语义层。

#### 主要工作

- 新建 `codeintel/`
- 第一阶段接入 LSP：
  - definition
  - references
  - document symbols
  - workspace symbols
  - diagnostics
- 定义 `CodeIntelProvider`
- 为 codeintel 结果建立统一返回结构，避免直接拼 prompt 文本
- 与 `ExecutionContext` 绑定 workspace
- 明确 fallback：
  - codeintel 可用时优先符号级
  - 不可用时退回 `FileRead/Grep/Glob`

#### 建议落点

- `codeintel/provider.py`
- `codeintel/models.py`
- `codeintel/lsp/`
- `Tool/builtin/lsp_tool.py` 或拆分 symbol/diagnostics 工具
- `runtime/context.py`

#### 验收标准

- 在真实仓库里完成定义查询、引用查询、诊断查询
- code agent 能优先用 symbol 级工具而不是盲搜文件
- 至少有一组集成测试验证 fallback 路径

### Phase E：Hooks / Guardrails + Tool Protocol v2

目标：把权限系统之外的内容级控制补齐，并完成工具协议升级。

#### 主要工作

- 新建 `core/hooks/`
- 新建 `core/guardrails/`
- 提供默认 hook 点：
  - `before_llm_request`
  - `after_llm_response`
  - `before_tool_use`
  - `after_tool_use`
  - `before_compaction`
  - `after_session_restore`
- Tool 协议补齐字段：
  - `side_effect_level`
  - `resource_scope`
  - `visibility_scope`
  - 同名冲突策略
- 明确 `resident / runtime / turn` 生命周期与冲突行为
- 把 `ephemeral_context` 纳入：
  - trace
  - compaction
  - session restore

#### 建议落点

- `core/hooks/`
- `core/guardrails/`
- `Tool/BaseTool.py`
- `Tool/ToolRegistry.py`
- `core/providers/tool_schema.py`

#### 验收标准

- hooks 能阻断或改写执行，不只是观察
- 工具协议可以稳定导出给多 provider
- runtime tool / turn tool / resident tool 的行为边界清晰

### Phase F：MCP Engineering

目标：把 MCP 从“能用”补到“框架一等扩展面”。

#### 主要工作

- 增加：
  - `mcp/connection_manager.py`
  - `mcp/auth.py`
  - `mcp/cache.py`
  - `mcp/policy.py`
- MCP server 纳入权限系统和 session/runtime 生命周期
- 增加 capability snapshot
- 明确 tool/resource/prompt 的统一来源标识
- 支持连接状态、重试、缓存失效、错误分类

#### 建议落点

- `mcp/`
- `core/permissions/`
- `runtime/context.py`
- `core/agent.py`

#### 验收标准

- MCP 连接状态、权限、缓存都可管理
- session restore 后能重建可恢复的 MCP runtime 状态
- builtin / skill / MCP 三类扩展来源可清晰区分

### Phase G：SDK 收口与通用 Agent 能力整理

目标：把框架正式收口成可发布 SDK。

#### 主要工作

- 增加 `pyproject.toml`
- 明确公共 API 边界
- 梳理安装方式与 extras
- 重整文档：
  - `framework API`
  - `product-like examples`
- 重整示例：
  - 单 agent
  - 多 agent
  - code agent
  - MCP
- 统一 `memory / rag / multimodal` 接入方式，挂到 runtime/context/tool/skill 抽象下

#### 建议落点

- `pyproject.toml`
- `docs/`
- `example/`
- `memory/`
- `rag/`

#### 验收标准

- 有稳定可安装的 SDK 入口
- 上层产品不必依赖内部文件布局
- 关闭 codeintel/team/task 时，基础 agent 仍能工作

## 推荐执行顺序

如果按当前代码状态推进，我建议严格按下面顺序做：

1. `Phase C`
2. `Phase D`
3. `Phase E`
4. `Phase F`
5. `Phase G`

这比旧计划更贴近当前现实，因为协作闭环已经补齐，当前最阻塞框架继续演进的是 runtime lifecycle 的长期运行与恢复边界。

## 本轮之后的最小任务包

如果只做接下来一轮最有价值的工作，建议先完成下面这个最小任务包：

1. 引入 `SessionRestoreReport` 和 runtime restore report
2. 明确 background agent / worktree / mailbox 的降级恢复语义
3. 给 runtime / session restore 增加结构化告警和清理策略
4. 补 lifecycle 集成测试与恢复文档

这是当前阶段性价比最高的一步，因为它会把 EasyAgent 从“协作闭环可用”推进到“长时间运行与恢复边界可信”。
