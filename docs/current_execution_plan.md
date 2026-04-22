# EasyAgent 当前执行计划

## 目标

这份计划基于当前仓库代码状态，而不是基于 `docs/walkthrough.md` 里的初始路线图重新从零排期。

框架定位保持不变：

- `code agent` 是一等公民
- 同时保留构建通用 agent 的抽象能力
- 优先补运行时闭环、恢复闭环、扩展协议，再补能力层
- 不优先做 CLI/TUI/voice 这类产品层能力

更新说明（2026-04-22）：

- `Phase A` 已完成：`AgentGet / AgentList / AgentWait / AgentStop`、后台 handle 语义、`completion records` 已补齐
- `Phase B` 已完成：`MailboxRead / MailboxAck`、message 生命周期、mailbox 自动注入 prompt、协作消费闭环已补齐
- `Phase C` 已完成：`SessionRestoreReport`、runtime/worktree restore report、`BaseAgent.close()` 生命周期收口已补齐
- `Phase D` 已完成：`codeintel/`、LSP stdio provider、codeintel tools、fallback 协议已补齐
- `Phase E` 已完成：`core/hooks/`、`core/guardrails/`、Tool Protocol v2、registry 冲突策略、`ephemeral_context` trace/restore 接线已补齐
- `Phase F` 已完成：`mcp/connection_manager.py`、`mcp/auth.py`、`mcp/cache.py`、`mcp/policy.py`、MCP runtime surface、session restore/close report 接线已补齐，说明见 `docs/phasef_mcp_engineering.md`
- `Phase G` 已完成：`pyproject.toml`、`easyagent/` 公共 SDK 门面、`docs/framework_api.md`、`example/README.md`、SDK 示例已补齐，说明见 `docs/phaseg_sdk_release.md`
- `Phase H` 已完成：`codeintel/cache.py`、workspace cache、offline symbol snapshot、`CodeIntelCacheStatus / CodeIntelPrewarmWorkspace`、codeintel runtime session restore 已补齐，说明见 `docs/phaseh_codeintel_workspace_cache.md`
- `Phase I` 已完成：`observability/recorder.py`、agent/llm/tool 统一观测、summary/recent events/trace summary、session restore、`easyagent.observability` 已补齐，说明见 `docs/phasei_observability_metrics.md`
- 当前下一阶段进入增强支线：`benchmark/exporter` 与更细的 `codeintel provider strategy`

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
- hooks / guardrails 已具备：
  - `core/hooks/`
  - `core/guardrails/`
  - `before_llm_request / after_llm_response / before_tool_use / after_tool_use / before_compaction / after_session_restore`
- Tool Protocol v2 已具备：
  - `side_effect_level`
  - `resource_scope`
  - `visibility_scope`
  - `ToolRegistry` 同名冲突策略
  - `ephemeral_context` 进入 trace 与 pending/session restore
- MCP engineering 已具备：
  - `mcp/auth.py`
  - `mcp/policy.py`
  - `mcp/cache.py`
  - `mcp/connection_manager.py`
  - MCP runtime export/restore
  - MCP runtime 进入 session restore 与 close report
  - ToolRegistry runtime surfaces: `mcp_manager` / `mcp_hub`
  - permission matcher 支持 `mcp_servers`
- code intelligence 已具备 LSP v1：
  - `codeintel/`
  - `LSPCodeIntelProvider`
  - `CodeIntelStatus / CodeIntelCacheStatus / CodeIntelPrewarmWorkspace / FindDefinition / FindReferences / GetDocumentSymbols / GetWorkspaceSymbols / GetDiagnostics`
  - 已具备 workspace cache、offline symbol snapshot、query cache fallback
  - session restore 后可重建 codeintel runtime 与 cache
- provider schema adapter 已落地：`core/providers/tool_schema.py`

### 已完成但只到 MVP 的能力

这些模块已经存在，但还没有形成最终闭环：

- `AgentTool` 能启动同步/后台子 agent，也能挂 team/worktree
- `AgentRuntimeManager` 能保存 handle、mailbox、team assignment，并支持 export/restore
- `SubagentManager` 能同步执行、后台执行、保存 output file
- runtime/team 状态可以进入 session restore
- codeintel 已具备 LSP v1 + workspace cache/offline snapshot，但更强的跨语言优化和更细的 provider 策略还没做完

### 当前最明显的缺口

这些缺口会直接限制 EasyAgent 继续长成 Claude Code 风格 code agent：

1. observability 主线已经补齐，但还可以继续增强

- 统一的 token / cost / error / tool metrics 聚合已落地
- `get_observability_summary()`、`get_recent_observability_events()`、`get_trace_summary()` 已可直接使用
- 后续增强重点变成 benchmark exporter、外部 metrics sink 和更细的多 agent/MCP/codeintel 分桶

2. SDK/package 边界已完成第一版收口，但后续仍可继续增强

- `pyproject.toml` 已落地
- 公共 API 门面 `easyagent/` 已落地
- `docs/framework_api.md` 和 `example/README.md` 已建立稳定入口索引
- 后续主要是继续细化兼容边界，而不是从零收口

3. code intelligence 还有增强空间

- 现在已具备 workspace cache 与 offline snapshot
- 但跨语言优化、批量预热策略和更细的 provider 选择仍可继续增强

## 新的优先级原则

`walkthrough.md` 里的旧顺序是合理的长期路线，但按当前代码状态，优先级需要调整成：

1. 先继续增强 observability：
   - benchmark exporter
   - 外部 metrics sink
2. codeintel 后续增强保留在并行支线：
   - 更细的跨语言优化
   - 更细粒度的 provider 策略
   - 更高效的批量预热

原因很简单：

- 现在多 agent 协作、runtime restore report、LSP v1、hooks/guardrails、Tool Protocol v2、MCP engineering 和 SDK 收口都已经跑通
- 当前最值得继续增强的是 observability exporter/benchmark 与更细粒度的 codeintel provider 策略
- codeintel 现在已经进入“可持续增强”阶段，而不是当前主线上最阻塞的缺口

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

状态：已完成核心实现，阶段说明见 `docs/phased_codeintel_lsp_v1.md`

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
- 让 `BaseAgent.close()` 收口 codeintel manager 生命周期

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

状态：已完成，阶段说明见 `docs/phasee_hooks_guardrails_tool_protocol_v2.md`

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

状态：已完成，阶段说明见 `docs/phasef_mcp_engineering.md`

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

状态：已完成，阶段说明见 `docs/phaseg_sdk_release.md`

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

### Phase H：CodeIntel Workspace Cache + Offline Snapshot

目标：把 codeintel 从“只依赖实时 LSP”升级成“实时 provider + 离线 cache/snapshot”双层结构。

状态：已完成，阶段说明见 `docs/phaseh_codeintel_workspace_cache.md`

#### 主要工作

- 新增 `codeintel/cache.py`
- 增加 `WorkspaceCodeIntelCache`
- 增加 `CodeIntelManager.prewarm_workspace()` 与 `get_cache_status()`
- 增加 `CodeIntelCacheStatus`、`CodeIntelPrewarmWorkspace`
- 为 `document symbols / diagnostics / definition / references` 增加 cache fallback
- 为 `workspace symbols` 增加 offline symbol snapshot fallback
- 让 `codeintel_runtime` 进入 session snapshot 与 restore report

#### 建议落点

- `codeintel/cache.py`
- `codeintel/manager.py`
- `Tool/builtin/codeintel_tools.py`
- `core/agent.py`

#### 验收标准

- `prewarm -> cache status -> offline symbol fallback` 全链路可验证
- provider 不可用时，已缓存的 `workspace symbols / document symbols / diagnostics / definition / references` 能按能力边界回退
- session restore 后 codeintel cache 仍可用

### Phase I：Observability Metrics + Trace Summary

目标：把 EasyAgent 的观测层从零散 callback 指标升级成正式 runtime 能力。

状态：已完成，阶段说明见 `docs/phasei_observability_metrics.md`

#### 主要工作

- 新增 `observability/recorder.py`
- 增加 `BaseObservabilityRecorder` 与 `InMemoryObservabilityRecorder`
- 为 plain / tool 的同步、异步、流式调用链补齐 agent run / llm request / tool execution 观测
- 增加 `get_observability_summary()`、`get_recent_observability_events()`、`get_trace_summary()`
- 让 `observability_state` 进入 session snapshot 与 restore
- 增加 `easyagent.observability`

#### 建议落点

- `observability/recorder.py`
- `core/agent.py`
- `agent/components/invocation_runner.py`
- `agent/components/tool_loop_engine.py`
- `easyagent/observability.py`

#### 验收标准

- plain / tool 的同步、异步、流式主链都有统一观测
- summary 能聚合 token / cost / error / tool metrics
- trace summary 能按 turn 输出 llm/tool 概览
- session restore 后 observability state 仍可读取

## 推荐执行顺序

如果按当前代码状态推进，我建议严格按下面顺序做：

1. observability 增强支线
   - benchmark exporter
   - 外部 metrics sink
2. codeintel 增强支线
   - 更细粒度的 provider 策略
   - 更高效的批量预热

这比旧计划更贴近当前现实，因为协作闭环、runtime restore report、LSP v1、workspace cache/offline snapshot、hooks/guardrails + Tool Protocol v2、MCP engineering，以及 observability 主线都已经补齐，接下来更适合做增强型支线。

## 本轮之后的最小任务包

如果只做接下来一轮最有价值的工作，建议先完成下面这个最小任务包：

1. 增加 benchmark exporter
2. 增加外部 metrics sink
3. 继续增强 codeintel provider strategy
4. 为增强支线补契约测试与真实 example

这是当前阶段性价比最高的一步，因为当前框架已经有 runtime、restore、codeintel cache、observability、hooks、Tool Protocol v2、MCP engineering 和 SDK 收口，下一步更适合做导出层和增强策略层。
