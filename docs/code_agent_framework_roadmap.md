# EasyAgent 面向 Code Agent 的框架路线图

本文档定义 EasyAgent 下一阶段的建设目标：不是直接复制 Claude Code 的终端产品形态，而是把 EasyAgent 补齐为一个足以承载 Claude Code 风格 code agent 的通用框架。

这里的“框架”含义是：

- 提供稳定的 runtime、protocol、policy、state management 和 extension points
- 允许上层产品以 CLI、IDE 插件、Web UI、远程服务等不同形式接入
- 避免把产品特有的交互形式直接写死在框架内核里

---

## 1. 目标与边界

### 1.1 最终目标

EasyAgent 应该能够支撑上层实现以下能力：

- 在代码仓库中进行安全的读写、搜索、执行和变更管理
- 支持 plan/execute 两阶段工作流
- 支持长任务、后台任务、任务恢复与多 agent 协作
- 支持代码智能能力，而不是只依赖文件级 grep/read/edit
- 支持基于规则和策略的权限控制
- 支持 MCP、Skill、内置 Tool 的统一扩展模型
- 支持会话恢复、上下文压缩、审计和可观测性

### 1.2 当前不作为框架优先项的内容

以下能力重要，但应优先放到“基于 EasyAgent 构建的产品层”中：

- Slash commands
- 终端 TUI / Web UI
- Voice / Vim mode / 移动端交互
- 品牌化输出样式
- 具体平台的账号系统与分发逻辑

这些能力需要框架为其提供接口，但不应驱动框架内核优先设计。

---

## 2. EasyAgent 当前基础

从现有代码看，EasyAgent 已经具备不少重要基础：

- 多类 Agent：`BasicAgent`、`ReactAgent`、`PlanningAgent`、`ConversationalAgent`、`StructuredOutputAgent`
- 工具系统：`ToolSpec`、`ToolResult`、`ToolRegistry`
- 内置 coding tools：`FileRead`、`Glob`、`Grep`、`FileWrite`、`FileEdit`、`Bash`、`NotebookEdit`
- 低频交互工具：`AskUserQuestion`、`ExitPlanMode`
- subagent / worktree 初步能力：`AgentTool`、`EnterWorktree`、`ExitWorktree`
- 会话保存与恢复：`save_session()` / `load_session()`
- history compaction 与 context 管理
- MCP tools/resources/prompts 的基础桥接
- Skill 体系与 on-demand skill 挂载

因此，下一阶段重点不是“从零做 agent 框架”，而是：

1. 把已有能力抽象得更稳定
2. 把缺失的 runtime 子系统补齐
3. 把产品层特性背后的内核协议补齐

---

## 3. 总体设计原则

### 3.1 框架优先，不做产品硬编码

例如：

- 框架中应定义 `PermissionDecision`，而不是直接耦合某个终端确认框
- 框架中应定义 `TaskStore`，而不是把任务逻辑写死在某个 TODO 文本文件里
- 框架中应定义 `PlanModeState`，而不是把“计划模式”只做成一个单独工具

### 3.2 协议先行

上层产品可以变化，但以下协议需要稳定：

- Tool 协议
- Permission 协议
- Task 协议
- Session / resume 协议
- Agent collaboration 协议
- Code intelligence 查询协议

### 3.3 风险显式化

code agent 的关键问题不是“能不能做”，而是“在什么边界内安全地做”。框架必须原生表达：

- 文件系统风险
- 网络风险
- 外部副作用
- 长时间运行
- 后台执行
- 破坏性命令

### 3.4 Provider-neutral

框架核心协议不能被某个模型供应商 API 绑死。`OpenAI function schema` 可以是一个适配目标，但不能成为事实标准。

---

## 4. 能力缺口总览

如果目标是支撑 Claude Code 风格的 code agent，EasyAgent 还需要补齐以下九个核心模块：

1. 权限与执行策略层
2. plan / execute 模式机
3. 结构化任务系统
4. 多 agent runtime
5. 代码智能层
6. Tool 协议工程化
7. Hooks / Guardrails / Policy extensions
8. 会话恢复与长上下文治理增强
9. MCP 生命周期管理

横向支撑模块还包括：

10. 可观测性与评测
11. 工程化打包与稳定 SDK 边界

下面按模块展开。

---

## 5. 模块设计

### 5.1 权限与执行策略层

#### 作用

为上层 code agent 提供统一的安全边界，决定：

- 哪些工具可以直接运行
- 哪些调用需要用户确认
- 哪些调用在 plan 模式只能规划不能执行
- 哪些目录、命令、网络访问是允许的
- 哪些规则只在当前 session 生效，哪些可持久化

#### 当前问题

当前 `ToolSpec` 主要只有：

- `read_only`
- `destructive`
- `requires_confirmation`

这对简单工具足够，但不足以支撑 code agent 的真实场景。缺少：

- 更细粒度的风险类型
- mode-aware 权限判断
- rule-based allow/deny
- 路径级、命令级、网络级策略
- session 级临时授权

#### 建议新增抽象

```python
class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "accept_edits"
    DONT_ASK = "dont_ask"
    BYPASS = "bypass"


class RiskCategory(str, Enum):
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    SHELL = "shell"
    NETWORK = "network"
    PROCESS = "process"
    MCP = "mcp"
    SIDE_EFFECT = "side_effect"


class PermissionRule(BaseModel):
    behavior: Literal["allow", "deny", "ask"]
    tool_name: str
    matcher: dict[str, Any] = Field(default_factory=dict)
    source: Literal["user", "project", "session", "policy"]
```

#### 建议文件

- `core/permissions/types.py`
- `core/permissions/rules.py`
- `core/permissions/engine.py`
- `core/permissions/context.py`
- `core/permissions/store.py`

#### 最低可用目标

- 能按 tool name + 风险类型做 allow/deny/ask
- 能表达 session 级权限规则
- 能在 plan mode 阻止高风险执行
- 能对文件路径与 shell 命令做策略匹配

---

### 5.2 Plan / Execute 模式机

#### 作用

为 agent 提供显式的执行阶段语义：

- `plan`：只做分析、拆解、询问信息、生成方案
- `execute`：执行工具、修改代码、运行命令

上层产品才能稳定构建：

- “先出计划再执行”
- “退出计划模式时声明允许的执行类别”
- “恢复 session 后仍保持正确模式”

#### 当前问题

当前已有 `ExitPlanMode` 工具，但它只是一个中断信号，不是完整模式系统。缺少：

- `EnterPlanMode`
- mode state 的持久化
- 模式切换时的权限重载
- 退出 plan 后的验证流程

#### 建议新增抽象

```python
class AgentExecutionMode(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"


class PlanModeState(BaseModel):
    mode: AgentExecutionMode
    entered_at: datetime | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    exit_requested: bool = False
```

#### 建议文件

- `core/execution_mode.py`
- `agent/components/mode_controller.py`
- `Tool/builtin/enter_plan_mode.py`
- `Tool/builtin/exit_plan_mode.py`

#### 最低可用目标

- Agent 内部有明确 mode state
- Tool 执行前会检查当前 mode
- mode state 可进入 session snapshot
- `AskUserQuestion`、`ExitPlanMode` 能和 mode state 真正联动

---

### 5.3 结构化任务系统

#### 作用

支撑长任务、后台任务、恢复继续做、多 agent 协作。

与文本型 TODO 的区别在于：任务是有状态、可查询、可演进的对象。

#### 建议任务模型

```python
class TaskStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskRecord(BaseModel):
    task_id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.OPEN
    owner: str | None = None
    parent_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
```

#### 需要的接口

- `TaskCreate`
- `TaskGet`
- `TaskUpdate`
- `TaskList`
- 可选：`TaskDelete`、`TaskLink`

#### 建议文件

- `task/models.py`
- `task/store.py`
- `task/service.py`
- `Tool/builtin/task_create.py`
- `Tool/builtin/task_get.py`
- `Tool/builtin/task_update.py`
- `Tool/builtin/task_list.py`

#### 最低可用目标

- 支持 SQLite store
- 能被 subagent 和 orchestrator 共享
- 能写入 session snapshot 或独立 task db
- 上层产品无需自己维护 TODO 文本协议

---

### 5.4 多 Agent Runtime

#### 作用

让 EasyAgent 从“可委派”升级成“可持续协作”。

不仅要能启动子 agent，还要支持：

- 后台运行
- agent 状态查询
- agent 间消息
- team / mailbox
- worktree 隔离
- 共享 task / shared context

#### 当前问题

当前已有：

- `AgentTool`
- `SubagentManager`
- `orchestrator/sequential.py`
- `orchestrator/supervisor.py`
- `orchestrator/group_chat.py`

但还缺：

- team 抽象
- mailbox / send message
- 统一运行状态模型
- 背景任务的可持续状态机

#### 建议新增抽象

```python
class AgentHandle(BaseModel):
    agent_id: str
    name: str
    status: Literal["running", "waiting", "completed", "error", "stopped"]
    workspace_root: str | None = None


class TeamHandle(BaseModel):
    team_id: str
    name: str
    members: list[str] = Field(default_factory=list)
```

#### 建议文件

- `runtime/agents/models.py`
- `runtime/agents/manager.py`
- `runtime/agents/mailbox.py`
- `runtime/teams/manager.py`
- `Tool/builtin/send_message.py`
- `Tool/builtin/team_create.py`
- `Tool/builtin/team_delete.py`

#### 最低可用目标

- 子 agent 可查询、可恢复、可停止
- agent 间支持结构化消息
- team 可以绑定一组 agent handle
- task 系统能和 agent runtime 打通

---

### 5.5 代码智能层

#### 作用

让 code agent 对代码库的理解从“文件级操作”升级到“符号级操作”。

#### 当前问题

目前主要依赖：

- `FileRead`
- `Glob`
- `Grep`
- `FileEdit`

这适合小仓库和简单任务，但在大仓库中会遇到明显上限：

- 无法高效查找 symbol 定义和引用
- 无法读取 LSP diagnostics
- 无法做跨文件语义级补全和定位

#### 目标能力

- `find_definition`
- `find_references`
- `get_document_symbols`
- `get_workspace_symbols`
- `get_diagnostics`
- 可选：代码索引 / embedding / symbol graph

#### 建议文件

- `codeintel/base.py`
- `codeintel/lsp_manager.py`
- `codeintel/indexer.py`
- `codeintel/models.py`
- `Tool/builtin/lsp_tool.py`

#### 最低可用目标

- 先接入 LSP diagnostics 和 symbol 查询
- 再做独立索引层，补足离线查询和跨语言支持

---

### 5.6 Tool 协议工程化

#### 作用

把现有 Tool 模块从“能注册和执行工具”升级成“框架级稳定协议”。

#### 当前问题

已有 `ToolSpec + ToolResult + ToolRegistry` 主干，但还缺：

- provider-neutral schema
- 更细风险语义
- 工具生命周期冲突检测
- turn/runtime/resident 工具更明确的装卸规则
- 临时上下文在 compaction / resume 后的稳定语义

#### 建议增强点

1. 风险模型扩展  
新增 `risk_categories`、`side_effect_level`、`resource_scope`

2. schema 适配层  
统一导出：
- OpenAI tools
- Anthropic tool schema
- Google function declarations
- MCP capability descriptors

3. 生命周期管理  
显式区分：
- `resident`
- `runtime`
- `turn`
- `ephemeral_result_context`

4. 冲突管理  
同名工具覆盖必须可配置，禁止静默覆盖

#### 建议文件

- `Tool/schema_adapters/`
- `Tool/lifecycle.py`
- `Tool/risk.py`
- `Tool/conflicts.py`

#### 最低可用目标

- ToolRegistry 对同名覆盖有显式策略
- ToolResult 的 `ephemeral_context` 可进入 trace 和 session snapshot
- 输出 schema 不再只依赖 OpenAI 风格

---

### 5.7 Hooks / Guardrails / Policy Extensions

#### 作用

允许框架用户在关键阶段插入自定义策略。

这类扩展点对 code agent 非常重要，因为很多组织级需求不会写死在内核中：

- secret 扫描
- prompt injection 检测
- 命令黑名单
- 结果过滤
- PRD / coding policy 注入

#### 当前问题

现有 `callbacks` 更偏 logging / metrics，不足以承担“可阻断、可修改输入输出”的 hook 语义。

#### 建议分层

- `callbacks`：只观察，不阻断
- `hooks`：可改写输入输出，可阻断
- `guardrails`：一组通用 hook/policy 实现

#### 建议 hook 点

- `before_llm_request`
- `after_llm_response`
- `before_tool_use`
- `after_tool_use`
- `before_compaction`
- `after_session_restore`

#### 建议文件

- `core/hooks/base.py`
- `core/hooks/manager.py`
- `core/guardrails/prompt_injection.py`
- `core/guardrails/secret_scanner.py`
- `core/guardrails/tool_policy.py`

#### 最低可用目标

- hook 能读取和改写 tool input
- hook 能阻止危险 tool 执行
- hook 能附加审计信息到 trace

---

### 5.8 会话恢复与长上下文治理增强

#### 作用

支撑真实 code agent 的长生命周期工作流：

- 中断恢复
- 跨天继续
- 工具中断后恢复
- 压缩后保持关键状态

#### 当前能力

已有：

- `save_session()`
- `load_session()`
- `compact_history()`
- `compact_persistent_history_if_needed()`

#### 还需要补的点

- tool interruption state 的稳定恢复
- runtime tool context 的恢复协议
- 更细的 compaction boundary / preserved tail 语义
- 恢复时的版本兼容策略

#### 建议文件

- `db/session_schema.py`
- `core/session_restore.py`
- `context/compaction_boundary.py`
- `context/compaction_protocol.py`

#### 最低可用目标

- 被中断的 tool loop 可恢复
- 临时 skill/tool context 恢复规则明确
- session schema 支持版本迁移

---

### 5.9 MCP 生命周期管理

#### 作用

把 MCP 从“能调用远程工具”升级成“可稳定运行的扩展面”。

#### 当前能力

已有：

- MCP tools 桥接
- MCP resources
- MCP prompts -> skill

#### 还需要补的点

- auth / OAuth
- server 级权限控制
- 自动重连
- 能力缓存
- 失效策略
- 统一展示本地 tools / MCP tools / skill tools 的来源边界

#### 建议文件

- `mcp/connection_manager.py`
- `mcp/auth.py`
- `mcp/cache.py`
- `mcp/policy.py`
- `mcp/discovery.py`

#### 最低可用目标

- MCP server 具备连接状态和错误分类
- 支持 capability snapshot 缓存
- 支持 server/tool 级权限策略

---

### 5.10 可观测性与评测

#### 作用

让框架使用者能系统评估一个 code agent 是否真的在变好。

#### 需要记录的东西

- token 使用
- 成本
- tool 调用链
- tool 成功率
- 权限拒绝原因
- compaction 发生频率
- session 恢复成功率

#### 建议文件

- `core/tracing.py`
- `core/metrics.py`
- `eval/benchmarks/`
- `eval/scenarios/`

#### 最低可用目标

- 每次 agent invoke 能产出结构化 trace
- trace 中可看到 `Agent -> Tool -> LLM -> Tool -> Final`
- 能离线跑一批 coding task benchmark

---

## 6. 推荐目录演进

建议在保持当前结构尽量稳定的前提下，新增以下目录：

```text
EasyAgent/
├── codeintel/
├── core/
│   ├── permissions/
│   ├── hooks/
│   ├── guardrails/
│   └── session/
├── runtime/
│   ├── agents/
│   ├── teams/
│   └── tasks/
├── task/
├── eval/
└── mcp/
    ├── auth.py
    ├── cache.py
    ├── policy.py
    └── connection_manager.py
```

说明：

- 若不希望新增 `runtime/` 顶层目录，也可以把相关实现继续放在 `Tool/runtime/` 与 `orchestrator/` 下
- 但从长期维护看，`tool runtime` 和 `agent runtime` 最终应拆开

---

## 7. 分阶段实施顺序

### Phase 1: 先把内核跑稳

目标：

- 权限系统 V1
- plan / execute 模式机
- 结构化任务系统 V1
- Tool 协议工程化 V1

验收标准：

- agent 能显式运行在 `plan` 或 `execute`
- 危险工具在不同 mode 下表现不同
- 任务可创建、查询、更新、列出
- session 级权限规则可生效

### Phase 2: 补齐 code agent 核心体验

目标：

- 多 agent runtime V1
- send message / team 能力
- LSP / diagnostics / symbol 查询
- session 恢复增强

验收标准：

- 子 agent 可后台运行、查询、停止
- team 可维护成员列表
- symbol 级查询可用于真实代码仓库
- 中断后的长任务可恢复

### Phase 3: 补齐扩展与稳定性

目标：

- hooks / guardrails
- MCP lifecycle
- tracing / metrics

验收标准：

- hook 可拦截危险 tool
- MCP 连接状态可管理
- trace 可用于问题排查和 benchmark

### Phase 4: 框架发布与 SDK 边界固化

目标：

- `pyproject.toml`
- 稳定 public API
- 文档与示例分层
- benchmark baseline

验收标准：

- 外部使用者能只依赖稳定 API 构建自己的 code agent 产品

---

## 8. 建议优先级

按“框架价值 / 对上层 code agent 影响”排序：

1. 权限与执行策略层
2. plan / execute 模式机
3. 结构化任务系统
4. 多 agent runtime
5. 代码智能层
6. Tool 协议工程化
7. Hooks / Guardrails
8. 会话恢复与长上下文治理增强
9. MCP 生命周期管理
10. 可观测性与评测

---

## 9. 不建议优先投入的方向

如果目标仍然是“框架”，以下方向不建议先做：

- 完整 CLI command 体系
- 终端 UI 样式系统
- voice mode
- vim mode
- 移动端交互
- 品牌化产品 packaging

原因不是它们不重要，而是这些属于上层产品问题，不能替代框架能力建设。

---

## 10. 最小成功标准

当以下条件同时满足时，可以认为 EasyAgent 已具备“Claude Code 风格 code agent 框架”的基础形态：

1. 有清晰的 permission engine，能表达 allow / deny / ask / mode-aware policy
2. 有明确的 `plan` 与 `execute` 模式机，状态可恢复
3. 有结构化 task system，不再依赖 TODO 文本模拟任务
4. 有可持续运行的 subagent / team runtime
5. 有 symbol / diagnostics 级 code intelligence
6. Tool 协议对 provider-neutral schema、risk semantics、lifecycle 有稳定抽象
7. session restore、compaction、tool interruption 三者之间协议一致
8. MCP 被视为一等扩展面，而不是临时桥接
9. trace / metrics / benchmark 能支撑持续迭代

---

## 11. 后续建议

建议先不要试图“一次对齐 Claude Code 所有表层功能”，而是把注意力集中在：

- runtime
- protocol
- policy
- code intelligence

一旦这四层打稳，上层产品无论是 CLI 还是 IDE 插件，都会容易很多；反过来，如果先做命令、界面和产品壳子，后续大概率会反复返工内核。

---

## 12. 建议下一步落地项

如果只选择一个短周期迭代，建议按下面顺序开工：

1. `core/permissions/`：做权限模式、规则、决策引擎
2. `core/execution_mode.py`：做 plan / execute 模式机
3. `task/`：做结构化任务模型和存储
4. `Tool` 与 `agent`：让 tool 执行统一走 permission + mode 检查

完成这四项后，再进入：

5. `runtime/agents/` 与 `runtime/teams/`
6. `codeintel/`
7. `core/hooks/` 与 `core/guardrails/`

这会是当前阶段性价比最高的路线。
