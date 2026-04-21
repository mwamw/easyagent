# EasyAgent 最终形态补充计划

## Summary

EasyAgent 的最终形态定位为：**以 Code Agent 为一等公民的通用 Agent 框架**。  
框架内核优先围绕 `runtime + protocol + policy + code intelligence + session lifecycle` 建设，确保未来能稳定承载 Claude Code 风格的 code agent；同时保留 `memory / rag / multimodal / workflow` 作为可选子系统，继续支撑更广义的通用 agent 构建。

默认架构目标：

- Code agent 是内核设计基准场景，不是后补插件
- 通用 agent 能力通过抽象层复用同一套 runtime、tool、session、policy 基础设施
- 现有 V2 memory、多模态、RAG、Skill、MCP 继续保留，但不主导主干架构
- 产品层能力如 CLI/TUI/Web/voice 不进入框架核心，只预留稳定接口

## Final Architecture

### 1. Runtime Kernel

- 新增统一的 `runtime` 层，承接当前分散在 `Tool/runtime`、`orchestrator`、`agent/components` 的运行时能力
- 运行时最终应包含四类核心对象：`AgentRuntime`、`ExecutionContext`、`TaskRuntime`、`SessionRuntime`
- `AgentRuntime` 负责单个 agent 的模式、状态、工具循环、暂停恢复、中断处理
- `TaskRuntime` 负责长任务、后台任务、任务状态流转、子任务依赖和多 agent 分派
- `SessionRuntime` 负责会话持久化、快照、恢复、版本迁移、compaction 边界管理
- `ExecutionContext` 统一表达当前工作目录、权限上下文、可见工具池、活跃 skills、MCP 连接和工作模式
- 当前 `orchestrator` 保留，但最终降级为 runtime 之上的 orchestration 策略层，而不是直接承担底层状态管理

### 2. Permission / Policy System

- 新增 `core/permissions/`，作为框架内核模块，而不是 ToolRegistry 内的轻量判断
- 引入一等公民概念：`PermissionMode`、`PermissionRule`、`PermissionDecision`、`RiskCategory`
- 工具权限判断最终必须支持：tool 级、路径级、命令级、网络级、server 级、session 级规则
- 支持至少五种模式：`default`、`plan`、`accept_edits`、`dont_ask`、`bypass`
- 所有工具执行统一经过 `permission engine`，而不是各工具自行判断
- `AskUserQuestion` 和未来的用户确认 UI/SDK 都只消费统一的 `PermissionDecision` 协议
- MCP、shell、filesystem、subagent、worktree 都必须接入同一套权限判定流程
- 框架默认提供规则匹配器，但允许上层产品替换策略来源和持久化方式

### 3. Plan / Execute Mode System

- 新增显式 `ExecutionMode` 模式机，作为 agent runtime 的基础状态，而不是仅靠 prompt 约定
- 模式最少包含 `plan` 与 `execute`，并允许后续扩展 `review`、`analysis`、`autonomous`
- 新增 `EnterPlanMode`，保留并重构 `ExitPlanMode`，二者都只负责模式转换协议，不直接耦合 UI
- tool loop 在 `plan` 模式下默认禁止高风险写操作和外部副作用执行
- mode state 必须进入 session snapshot，恢复后模式不丢失
- mode 切换后会重新计算当前可见工具池与允许动作集合
- `PlanningAgent` 后续应重构为 mode-aware agent，而不是与 `BasicAgent` 完全分裂的一套逻辑

### 4. Task System

- 新增结构化任务子系统，替代当前只有 `TodoWrite` 的轻量记录方式
- 最终公开接口至少包含：`TaskCreate`、`TaskGet`、`TaskUpdate`、`TaskList`
- 任务模型至少包含：`task_id`、`title`、`description`、`status`、`owner`、`parent_task_id`、`metadata`
- 状态至少包含：`open`、`in_progress`、`blocked`、`completed`、`cancelled`
- 子 agent、team、workflow、session restore 都统一依赖该任务系统，而不是各写一套状态文件
- `TodoWrite` 最终保留为用户可读摘要视图或兼容工具，不再作为 runtime 真正的任务源
- 默认存储先做 SQLite，接口抽象支持后续 Redis / Postgres

### 5. Multi-Agent Runtime

- 在现有 `AgentTool`、`SubagentManager`、`orchestrator` 基础上，升级为统一多 agent runtime
- 最终支持四类对象：`Subagent`、`BackgroundAgent`、`Team`、`Mailbox`
- 新增 `SendMessage` 作为 agent 间结构化消息工具
- 新增 `TeamCreate`、`TeamDelete`，使多个 agent 能被显式编组成 team
- 子 agent 状态必须可查询、可恢复、可中断、可统计，不仅是一次性运行结果
- worktree 隔离保留为 runtime 的一种 workspace strategy，而不是工具层孤立功能
- 支持共享 `TaskStore`、共享权限上下文、按 agent 粒度覆盖工具池
- 后续 DAG orchestration、workflow scripts、coordinator 模式都应建立在这一 runtime 之上

### 6. Code Intelligence Layer

- 新增 `codeintel/` 子系统，作为 code agent 的基础能力层
- 第一阶段以 LSP 为主，最终支持：definition、references、symbols、diagnostics、workspace search
- 第二阶段补充索引层，用于大仓库离线检索、跨语言支持、符号摘要和上下文裁剪
- 代码智能结果应进入统一 context pipeline，而不是直接拼 prompt 文本
- Tool 层最终暴露的不是“只有文件读写”，而是“文件工具 + 符号工具 + 诊断工具”
- `FileRead/Grep/Glob` 继续保留，作为底层 fallback 和通用 agent 的基础文件工具

### 7. Tool Protocol Engineering

- 保持 `ToolSpec + ToolResult + ToolRegistry` 作为主干，但扩展为稳定协议层
- 新增字段：`risk_categories`、`side_effect_level`、`resource_scope`、`visibility_scope`
- 生命周期最终显式区分：`resident`、`runtime`、`turn`、`ephemeral_context`
- `ToolRegistry` 必须支持同名冲突策略，禁止静默覆盖
- `ToolResult.ephemeral_context` 必须进入 trace、compaction 和 session rebuild 协议
- 新增 schema adapter 层，统一导出 OpenAI、Anthropic、Google、MCP 所需 schema
- 工具授权和工具 schema 生成解耦，避免把 provider 细节压进 Tool 本身

### 8. Hooks / Guardrails

- 保留 `callbacks` 作为观察层，不承担阻断语义
- 新增 `core/hooks/` 作为可改写输入输出、可阻断执行的扩展层
- 新增 `core/guardrails/` 提供默认策略实现，如 prompt injection 防护、secret 扫描、危险命令拦截
- hook 点至少覆盖：`before_llm_request`、`after_llm_response`、`before_tool_use`、`after_tool_use`、`before_compaction`、`after_session_restore`
- hooks 应返回统一结构，支持：允许、阻断、改写输入、附加审计信息
- 权限系统与 hooks 可以叠加，但职责分离：权限系统做规则判定，hooks 做策略扩展和内容级检查

### 9. Session / Compaction / Recovery

- 在现有 `save_session`、`load_session`、`compact_history` 基础上升级为版本化会话协议
- session snapshot 最终必须覆盖：mode state、task state、permission context、tool interruption、runtime mounts、replay history metadata
- compaction 需要显式边界对象，允许恢复后区分“压缩摘要”和“保留原始尾部”
- 恢复流程必须能检测缺失工具、缺失 skill、provider 不匹配和 schema 版本漂移
- 长任务恢复必须能从中断点继续，而不是仅恢复对话历史
- `trace_history` 与正式会话历史继续分层，但要有稳定的关联键，便于审计和重建

### 10. MCP as First-Class Extension Surface

- 现有 MCP tools/resources/prompts 保留，并升级为统一 capability surface
- 新增 `mcp/connection_manager.py`、`mcp/auth.py`、`mcp/cache.py`、`mcp/policy.py`
- MCP server 最终必须有：连接状态、错误分类、权限上下文、capability snapshot、缓存与失效策略
- MCP tools、resources、prompts 在框架内都要有统一来源标识，避免和 builtin tools、skills 混淆
- MCP prompt 最终可映射到 skill 或 command，但映射规则应显式配置，不做隐式魔法
- MCP 必须纳入统一 permission engine，而不是仅靠工具元信息中的 `requires_confirmation`

### 11. General-Agent Optional Subsystems

- `memory/V2`、`rag/`、多模态 perceptual 能力保留，并定位为可选能力包
- 这些子系统最终通过 `ContextSource`、`Skill` 或 `Tool` 挂接进运行时，而不反向主导核心 runtime
- 现有知识图谱、多模态、语义记忆仍可继续演进，但路线图上优先级晚于 runtime / permissions / codeintel
- 通用 agent 的能力建设优先放在“如何通过统一抽象接入”，而不是继续扩大孤立子系统

### 12. Packaging / Stable SDK

- 新增 `pyproject.toml`，明确公共安装方式与 extras
- 框架最终应有稳定 public API 边界，至少覆盖：Agent、Tool、Task、Permission、Session、MCP、Hooks
- `example/` 与 `docs/` 应分出“框架 API 示例”和“产品级示例”两类，避免混淆
- 后续 CLI 或产品层仓库应以 EasyAgent SDK 形式接入，而不是直接依赖内部文件布局

## Public APIs / Interfaces

框架最终需要新增或稳定以下接口族：

- `core.permissions`
  - `PermissionMode`
  - `RiskCategory`
  - `PermissionRule`
  - `PermissionDecision`
  - `PermissionEngine`
- `core.execution_mode`
  - `ExecutionMode`
  - `PlanModeState`
  - `ModeController`
- `task`
  - `TaskRecord`
  - `TaskStatus`
  - `TaskStore`
  - `TaskService`
- `runtime.agents`
  - `AgentHandle`
  - `BackgroundAgentHandle`
  - `MailboxMessage`
  - `SubagentManager` 重构版
- `runtime.teams`
  - `TeamHandle`
  - `TeamManager`
- `codeintel`
  - `CodeIntelProvider`
  - `DefinitionQuery`
  - `ReferenceQuery`
  - `DiagnosticRecord`
- `Tool`
  - 扩展后的 `ToolSpec`
  - 扩展后的 `ToolResult`
  - `SchemaAdapter`
- `core.hooks`
  - `HookDecision`
  - `HookManager`
  - `PreToolHook`
  - `PostToolHook`
- `core.session`
  - `SessionSnapshotV2`
  - `SessionMigration`
  - `SessionRestoreReport`
- `mcp`
  - `MCPConnectionManager`
  - `MCPCapabilitySnapshot`
  - `MCPPolicyContext`

这些接口需要尽量保持 provider-neutral、UI-neutral、product-neutral。

## Execution Plan

基于当前仓库状态，EasyAgent 已完成 `Phase 1` 的 MVP 级基础能力，但还没有达到“最终版”。
后续执行采用 `Code-Agent 优先`、`7 期细分` 的路线：先把内核补到可长期承载 code agent 的稳定版本，再完成多 agent runtime、代码智能、生命周期治理、MCP 工程化，最后收口为可发布的通用 Agent 框架。

这份执行计划遵循三个原则：

- 先补运行时主干，再补能力层和扩展层
- 每一期都必须有明确的接口落点和验收门槛
- “最终版”定义为：核心能力完整、恢复链路完整、扩展协议稳定、通用 agent 兼容、可作为独立 SDK 发布

### Phase 1A: Kernel Finalization

目标：把当前已做的权限、模式、任务三条主线从 MVP 补到稳定版。

- 权限系统补齐 `PermissionStore`、规则来源优先级、`accept_edits` 语义、`dont_ask/bypass` 完整行为，以及更明确的路径、命令、网络匹配边界
- plan/execute 模式补齐与 `AskUserQuestion`、`ExitPlanMode` 的联动，保证 mode 切换后工具池和权限视图一致
- 任务系统补齐 `TodoWrite -> Task summary view` 的降级关系，避免 TODO 和结构化任务双轨分裂
- session snapshot 升级到最小 `V2` 形态，至少稳定包含 `mode_state / permission_context / current_task_id / replay history metadata`
- 修复现有 session persistence 基线问题，保证当前仓库里的会话保存恢复测试口径统一

本期新增或稳定的接口：

- `core.permissions.PermissionStore`
- `core.session.SessionSnapshotV2` 最小版
- `task.TaskService` 作为任务唯一事实源

本期完成标准：

- 当前 Phase 1 新增能力不再依赖临时接线逻辑
- session、permission、task 三者能一致恢复
- `test_session_persistence` 全绿

### Phase 2: Runtime Core

目标：落地统一 runtime 主骨架，替代当前分散的 subagent、worktree、runtime 状态。

当前状态：已完成当前阶段目标，最终收口说明见 `docs/phase23_mailbox_collaboration_complete.md`。

- 建立 `runtime/agents/`，引入 `AgentHandle`、`BackgroundAgentHandle`、`MailboxMessage`、`CompletionRecord`、新版 `SubagentManager`
- 建立 `ExecutionContext`，统一表达工作目录、权限上下文、任务上下文、工具池、mode、worktree、MCP 可见性
- 把 `AgentTool` 从“单次委派工具”升级成“创建、查询、恢复 subagent handle 的 runtime 入口”
- worktree 不再作为孤立工具能力，而是挂到 `ExecutionContext` 里，供 subagent 共享与继承
- 统一 subagent 生命周期：创建、运行、查询、停止、恢复、销毁

本期新增或稳定的接口：

- `runtime.agents.AgentHandle`
- `runtime.agents.BackgroundAgentHandle`
- `runtime.agents.MailboxMessage`
- `runtime.agents.CompletionRecord`
- `runtime.ExecutionContext`

本期完成标准：

- 子 agent 可查询、可恢复、可中断
- worktree 和 subagent 共享同一 runtime context
- `AgentTool` 只做 runtime 派发，不再直接承担状态管理

### Phase 3: Collaboration Layer

目标：完成多 agent 协作层，而不只是 subagent 启动。

当前状态：已完成当前阶段目标，最终收口说明见 `docs/phase23_mailbox_collaboration_complete.md`。

- 落地 `runtime/teams/`，提供 `TeamHandle`、`TeamManager`
- 新增 `SendMessage`、`TeamCreate`、`TeamDelete`，让 agent 间能传结构化消息、编组、解组
- 新增 `MailboxRead`、`MailboxAck`，让 mailbox 成为可消费的协作协议
- mailbox 支持点对点消息、team 广播、task 关联消息
- task 系统与 agent runtime 打通：任务可绑定 owner agent、team、parent task
- 背景 agent 和 team 状态要能进入 session 恢复链路

本期新增或稳定的接口：

- `runtime.teams.TeamHandle`
- `runtime.teams.TeamManager`
- `Tool.SendMessage`
- `Tool.MailboxRead`
- `Tool.MailboxAck`
- `Tool.TeamCreate`
- `Tool.TeamDelete`

本期完成标准：

- agent 间消息是结构化对象，不靠 prompt 文本模拟
- mailbox 消息会进入子 agent 的执行上下文，并支持 read / ack 生命周期
- team 能稳定管理一组 agent handle
- task、agent、team 三者关系可查询、可恢复

### Phase 4: Code Intelligence v1

目标：让框架具备真正可用的 code agent 语义层。

- 落地 `codeintel/`，先以 LSP 为主，实现 definition、references、document/workspace symbols、diagnostics
- 增加 `CodeIntelProvider` 抽象，避免直接把 LSP 细节压进工具层
- 新增 codeintel 工具，但结果统一进入 context pipeline，而不是直接拼成 prompt 文本
- 保留 `FileRead/Grep/Glob` 作为 fallback；当 codeintel 不可用时，回退路径要明确
- 把 codeintel 与 `ExecutionContext` 绑定，支持 workspace 级索引视图

本期新增或稳定的接口：

- `codeintel.CodeIntelProvider`
- `codeintel.DefinitionQuery`
- `codeintel.ReferenceQuery`
- `codeintel.DiagnosticRecord`

本期完成标准：

- 在真实代码仓库中完成 symbol 查询和 diagnostics 查询
- LSP 不可用时 fallback 行为稳定
- code agent 默认优先使用符号级能力而不是文件级盲搜

### Phase 5: Lifecycle Hardening

目标：把框架从“能运行”升级到“能长期运行和恢复”。

当前状态：`SessionRestoreReport`、runtime/worktree restore report 与 `BaseAgent.close()` 生命周期收口已完成，说明见 `docs/phasec_restore_report_lifecycle.md`；`hooks / guardrails / trace 关联键 / observability` 仍待后续实现。

- 落地 `core/hooks/` 与 `core/guardrails/`，让内容级策略与规则级权限分层协作
- session、compaction、interruption 恢复协议升级，覆盖 tool interruption、runtime mounts、missing tools、provider drift
- 建立 `SessionRestoreReport`，恢复时显式报告降级、缺失、漂移，而不是静默跳过
- 建立 trace 关联键，打通 `trace_history`、tool result、compaction summary、session rebuild
- 加入最小 observability：token、工具调用链、错误类型、恢复结果

本期新增或稳定的接口：

- `core.hooks.HookManager`
- `core.hooks.HookDecision`
- `core.session.SessionRestoreReport`

本期完成标准：

- 长任务可以从中断点恢复
- session restore 失败或降级时有结构化报告
- hooks 可以阻断和改写执行，而不仅是观察

### Phase 6: Tool Protocol + MCP Engineering

目标：完成扩展协议的最终版，保证框架具备长期可扩展性。

- Tool 协议补齐 `side_effect_level`、`resource_scope`、`visibility_scope`、生命周期分层、同名冲突策略
- 引入 schema adapter，统一导出 OpenAI、Anthropic、Google、MCP 所需 schema
- `ToolResult.ephemeral_context` 纳入 trace、compaction、session restore 协议
- MCP 升级为 first-class extension surface，补齐 `connection_manager / auth / cache / policy / capability snapshot`
- MCP、builtin tool、skill 的来源标识统一，避免扩展面混乱

本期新增或稳定的接口：

- `Tool.SchemaAdapter`
- 扩展后的 `ToolSpec`
- 扩展后的 `ToolResult`
- `mcp.MCPConnectionManager`
- `mcp.MCPCapabilitySnapshot`
- `mcp.MCPPolicyContext`

本期完成标准：

- 工具协议对 provider-neutral 成立
- MCP 生命周期、权限、缓存、连接状态可管理
- 扩展系统不再依赖隐式约定

### Phase 7: General-Agent Consolidation + SDK Release

目标：把框架收口成“既支持 code agent，也支持通用 agent”的最终发布形态。

- 重新整理 `memory / rag / multimodal` 接入方式，统一挂到 runtime、context、tool、skill 抽象下
- 明确公共 API 边界，避免上层产品依赖内部文件布局
- 增加 `pyproject.toml`、extras、公共安装方式和发布结构
- 重构文档和 examples，区分“框架 API 示例”和“产品形态示例”
- 做最终兼容验收：关闭 codeintel、team、task 时，基础 `BasicAgent + Tool + Skill + Memory` 仍可工作

本期新增或稳定的接口：

- 稳定的 `Agent / Tool / Task / Permission / Session / MCP / Hooks` 公共 API
- 安装 extras 和 SDK 使用边界

本期完成标准：

- EasyAgent 可以作为独立 SDK 被外部项目接入
- Code agent 与通用 agent 场景都通过验收
- 文档与示例足以支撑二次开发

## Test Plan

- Kernel gate
  - 权限模式、规则、任务、session restore 全链路测试全绿
  - 旧 agent 类型恢复、模式切换、task 绑定不回归
- Runtime gate
  - subagent 创建、查询、停止、恢复
  - worktree 继承与 execution context 一致
- Collaboration gate
  - `SendMessage`、`TeamCreate`、`TeamDelete` 的结构化消息和 team 生命周期稳定
  - task、agent、team 绑定关系恢复正确
- Codeintel gate
  - definition、references、symbols、diagnostics 在真实仓库可用
  - LSP 不可用时 fallback 正常
- Lifecycle gate
  - interruption、compaction、restore report、provider drift、缺失工具等场景可验证
- Protocol/MCP gate
  - 多 provider schema 导出一致
  - MCP 断连、重连、权限拒绝、缓存失效都有明确行为
- Final GA gate
  - code agent 场景、通用 agent 场景、无 codeintel 场景都能通过验收

## Assumptions and Defaults

- 当前起点是：`Phase 1` MVP 已存在，后续计划从“补齐到最终版”开始，而不是从零开始
- 优先级采用 `Code-Agent 优先`，因此 `runtime / collaboration / codeintel` 会早于 `memory / rag / multimodal` 重构
- 分期采用 7 期细分，每一期必须先通过验收门槛，再进入下一期
- CLI、TUI、Web、voice 仍不纳入框架主线交付物，只保证框架为其预留稳定接口
