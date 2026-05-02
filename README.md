# EasyAgent 🚀

**一个轻量、透明、为真实工程和长程任务而生的 AI Agent 运行时框架。**

### 💡 为什么我们需要 EasyAgent？

在市面上已经有 LangChain、LlamaIndex 等优秀框架的今天，为什么还要重造轮子？

因为当我们真正尝试在本地构建类似 Devin 的高阶 Code Agent（如 [S4Code](https://github.com/mwamw/S4Code)）时，我们发现现有的工具链存在严重的**“过度封装”**与**“状态脆弱性”**。当 Agent 需要在你的真实代码仓库中执行长达数十步的阅读、修改、测试和提交操作时，你最需要的不再是一个快速搭建 Demo 的黑盒玩具，而是一个**极致可控的底层 Runtime**。

EasyAgent 放弃了臃肿的抽象层，回归工程本质，为你提供以下核心能力：

* **🔍 拒绝黑盒，极致的透明度：** 将 Tool Loop、Output Parser 与 Hooks 彻底解耦。内置 Trace Recorder 让你能精准追踪每一轮 Reasoning 和 Tool Result。Agent 为什么死循环？哪一步参数传错了？一目了然。
* **💾 原生状态管理与断点续跑：** 复杂的长程任务最怕在第 49 步因为网络波动或 Token 超载而崩溃。EasyAgent 独创 `ExecutionContext` 与会话快照，记录完整的工具调用链路。崩溃了？直接从错误点秒级重建上下文继续跑，告别 Token 浪费。
* **🛡️ “人在回路”的安全权限引擎：** 大模型绝不能在没有授权的情况下裸奔。内置首创的 `PermissionContext`，在涉及文件覆写、Shell 执行、Git 修改等高危动作前，原生支持 **Ask / Allow / Deny** 拦截策略，兼顾自动化与绝对安全。
* **🔌 干净的抽象与前沿协议支持：** 抹平 OpenAI、Anthropic、Google 御三家在 Tool Calling 和结构化输出上的底座差异，并且**原生支持 MCP (Model Context Protocol)** 接入，以最轻的姿态拥抱最新的工具链生态。

**简而言之：如果你只是想花 5 分钟写一个聊天的 Demo，请使用 LangChain；如果你想构建一个在真实文件系统里跑满 10 分钟不出错、高度可控且安全的工业级 Agent 工具，欢迎使用 EasyAgent。**
EasyAgent 是一个面向通用 Agent、Code Agent 和多 Agent 协作场景的 Python 框架。
它关注的不是“怎么调用一次模型”，而是“怎么把模型、工具、权限、会话、上下文、运行时和可观测性组装成可持续维护的 Agent 产品”。

这份 README 不是简介，而是整个框架文档体系的总入口。
如果你只读一个文件，请先读它；如果你要做真正的产品化集成，请顺着这里的主题索引继续读 `docs/` 下的专题手册。

## 1. 你可以用 EasyAgent 做什么

EasyAgent 适合构建：

- 单智能体对话助手
- 带工具调用的自动化助手
- 本地或远程运行的 Code Agent
- 带 Ask / Allow / Deny 权限模型的命令执行代理
- 带 Skill、MCP、CodeIntel、RAG、Memory 的工程助手
- 支持 Team / Task / Mailbox / background agent 的多 Agent 系统

## 2. 快速安装

核心安装：

```bash
pip install -e .
```

可选扩展：

```bash
pip install -e ".[mcp]"
pip install -e ".[rag]"
pip install -e ".[memory]"
pip install -e ".[dev]"
```

推荐：

- 只做工具型 Agent：核心安装即可
- 要接 MCP：加 `.[mcp]`
- 要接知识库或检索：加 `.[rag]`
- 要长期记忆：加 `.[memory]`
- 要跑完整测试：加 `.[dev]`

## 3. 30 秒最小 Agent

```python
from easyagent import BasicAgent, EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="test",
    model="qwen3.5-9b",
)

agent = BasicAgent(name="assistant", llm=llm)
print(agent.invoke("用一句话说明 EasyAgent 是什么。"))
```

## 4. 20 行最小 Code Agent

```python
from easyagent import BasicAgent, Config, EasyLLM
from easyagent.tools import ToolRegistry, register_filesystem_tools, register_shell_tools

llm = EasyLLM(
    provider="anthropic_native",
    api_key="test",
    base_url="http://127.0.0.1:5124",
    model="deepseek-v4-flash:zenmux:claude",
)

config = Config(tool_schema_mode="deferred")
registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root=".")
register_shell_tools(registry, workspace_root=".", expose_in_deferred=False)

agent = BasicAgent(
    name="code-agent",
    llm=llm,
    enable_tool=True,
    config=config,
    tool_registry=registry,
)
```

更完整的产品装配示例见：

- [Code Agent Product Quickstart](./docs/code_agent_product_quickstart.md)

## 5. 公共 SDK 导入方式

新项目建议统一从 `easyagent` 或 `easyagent.*` 导入，而不是直接依赖内部目录。

常见导入：

```python
from easyagent import BasicAgent, EasyLLM, Config
from easyagent.tools import ToolRegistry, register_filesystem_tools, register_shell_tools
from easyagent.permissions import PermissionContext, PermissionEngine, PermissionRule
from easyagent.callbacks import CallbackManager, StreamingCallback
from easyagent.prompting import PromptBlock, BasePromptComposer
from easyagent.reminders import RuntimeReminder, BaseRuntimeReminderSource
```

完整公共 API 索引见：

- [Framework API](./docs/framework_api.md)

## 6. 文档导航

### 6.1 入门与公共 API

- [Framework API](./docs/framework_api.md)
- [Config Reference](./docs/config_reference.md)
- [Agent Guide](./docs/agent_guide.md)
- [LLM Provider Guide](./docs/llm_provider_guide.md)
- [Code Agent Product Quickstart](./docs/code_agent_product_quickstart.md)

### 6.2 Tool 系统

- [Tool System Guide](./docs/tool_system_guide.md)
- [Builtin Tools Catalog](./docs/builtin_tools_catalog.md)
- [Tool Authoring Guide](./docs/tool_authoring_guide.md)
- [Deferred Tools Guide](./docs/deferred_tools_guide.md)

### 6.3 Prompt、Reminder、Skill

- [Prompt System Guide](./docs/prompt_system_guide.md)
- [Prompt Composer Guide](./docs/prompt_composer_guide.md)
- [Runtime Reminders Guide](./docs/runtime_reminders_guide.md)
- [Skill System Guide](./docs/skill_system_guide.md)

### 6.4 Context、Memory、Session

- [Memory System Guide](./docs/memory_system_guide.md)
- [Context And Compaction Guide](./docs/context_and_compaction_guide.md)
- [Session Restore Persistence Guide](./docs/session_restore_persistence_guide.md)

### 6.5 Runtime、Task、Permission、Hook、Callback

- [Runtime Collaboration Guide](./docs/runtime_collaboration_guide.md)
- [Tasks Guide](./docs/tasks_guide.md)
- [Permissions Guide](./docs/permissions_guide.md)
- [Hooks And Guardrails Guide](./docs/hooks_and_guardrails_guide.md)
- [Callbacks And Streaming Guide](./docs/callbacks_and_streaming_guide.md)

### 6.6 扩展能力

- [MCP Guide](./docs/mcp_guide.md)
- [CodeIntel Guide](./docs/codeintel_guide.md)
- [Observability And Cache Guide](./docs/observability_and_cache_guide.md)
- [RAG Guide](./docs/rag_guide.md)
- [Worktree Guide](./docs/worktree_guide.md)

## 7. 核心概念总览

### Agent

`BaseAgent` 是运行时基座，`BasicAgent` 是默认可用的通用实现。
Agent 负责把 LLM、Tool、Skill、History、Callback、Hook、Permission、Runtime 组装起来。

详见：

- [Agent Guide](./docs/agent_guide.md)

### EasyLLM / Provider

`EasyLLM` 是统一的模型访问层。
框架通过 provider 适配不同协议，例如 `openai`、`openai_responses`、`google_native`、`anthropic_native`。

详见：

- [LLM Provider Guide](./docs/llm_provider_guide.md)

### Tool

Tool 是 Agent 的执行能力。
EasyAgent 提供 Tool 基类、ToolRegistry、权限判定、确认中断、deferred schema、runtime/turn 可见性，以及大量 builtin tools。

详见：

- [Tool System Guide](./docs/tool_system_guide.md)
- [Builtin Tools Catalog](./docs/builtin_tools_catalog.md)
- [Tool Authoring Guide](./docs/tool_authoring_guide.md)

### Prompt / Runtime Reminder

EasyAgent 不把所有上下文都塞进 system prompt。
系统提示词、runtime reminder、on-demand expansion、dynamic tail 是不同层，目的包括更清晰的请求结构和更稳定的 cache 前缀。

详见：

- [Prompt System Guide](./docs/prompt_system_guide.md)
- [Runtime Reminders Guide](./docs/runtime_reminders_guide.md)

### Skill

Skill 是“能力包”，可以同时提供：

- prompt 片段
- tool
- context source
- 按需激活逻辑

详见：

- [Skill System Guide](./docs/skill_system_guide.md)

### Context / Memory / History Compaction

Context 模块负责“请求时附加上下文”，Memory 模块负责“长期或工作记忆”，Compaction 负责“当上下文过长时如何压缩历史”。

详见：

- [Memory System Guide](./docs/memory_system_guide.md)
- [Context And Compaction Guide](./docs/context_and_compaction_guide.md)

### Session

Session 模块负责持久化 agent 快照和消息历史，并提供恢复报告。

详见：

- [Session Restore Persistence Guide](./docs/session_restore_persistence_guide.md)

### Runtime / Team / Task / Mailbox

Runtime 负责 subagent、background task、team 协作、mailbox 广播和执行上下文管理。

详见：

- [Runtime Collaboration Guide](./docs/runtime_collaboration_guide.md)
- [Tasks Guide](./docs/tasks_guide.md)

### Permission / Hook / Callback

- Permission：决定某个工具调用是 Ask、Allow 还是 Deny
- Hook：在关键执行点阻塞、修改、放行 payload
- Callback：做非阻塞观测、UI 更新、日志采集

详见：

- [Permissions Guide](./docs/permissions_guide.md)
- [Hooks And Guardrails Guide](./docs/hooks_and_guardrails_guide.md)
- [Callbacks And Streaming Guide](./docs/callbacks_and_streaming_guide.md)

### MCP / CodeIntel / RAG / Worktree

这些是更偏产品增强的扩展层：

- MCP：远程工具与资源接入
- CodeIntel：定义、引用、诊断、符号
- RAG：知识库与检索增强
- Worktree：隔离执行环境

详见：

- [MCP Guide](./docs/mcp_guide.md)
- [CodeIntel Guide](./docs/codeintel_guide.md)
- [RAG Guide](./docs/rag_guide.md)
- [Worktree Guide](./docs/worktree_guide.md)

## 8. 模块集成顺序建议

如果你要从零做一个产品，推荐按这个顺序接模块：

1. `EasyLLM`
2. `BasicAgent`
3. `Config`
4. `ToolRegistry` + builtin tools
5. `PermissionContext` / `PermissionEngine`
6. `CallbackManager`
7. `PromptBlock` / runtime reminders
8. `SkillManager`
9. `SessionStore` / `ConversationStore`
10. `ObservabilityRecorder`
11. 视场景继续接 `Runtime` / `MCP` / `CodeIntel` / `Memory` / `RAG`

## 9. 最常见的三条产品路径

### 9.1 通用 Agent

适合问答、工具调用和基本会话。

重点阅读：

- [Agent Guide](./docs/agent_guide.md)
- [Tool System Guide](./docs/tool_system_guide.md)
- [Prompt System Guide](./docs/prompt_system_guide.md)

### 9.2 Code Agent

适合本地仓库分析、读写文件、shell、子 agent、worktree。

重点阅读：

- [Code Agent Product Quickstart](./docs/code_agent_product_quickstart.md)
- [Builtin Tools Catalog](./docs/builtin_tools_catalog.md)
- [Deferred Tools Guide](./docs/deferred_tools_guide.md)
- [Permissions Guide](./docs/permissions_guide.md)
- [Worktree Guide](./docs/worktree_guide.md)
- [CodeIntel Guide](./docs/codeintel_guide.md)

### 9.3 Multi-Agent 协作系统

适合 manager + worker、广播、后台子任务、结构化任务推进。

重点阅读：

- [Runtime Collaboration Guide](./docs/runtime_collaboration_guide.md)
- [Tasks Guide](./docs/tasks_guide.md)
- [Session Restore Persistence Guide](./docs/session_restore_persistence_guide.md)

## 10. 你最可能先问的几个问题

### 怎么自定义系统提示词

看：

- [Prompt System Guide](./docs/prompt_system_guide.md)
- [Prompt Composer Guide](./docs/prompt_composer_guide.md)

### 怎么自定义工具

看：

- [Tool Authoring Guide](./docs/tool_authoring_guide.md)
- [Tool System Guide](./docs/tool_system_guide.md)

### deferred tool 到底怎么工作

看：

- [Deferred Tools Guide](./docs/deferred_tools_guide.md)

### 怎么把 memory / session / callback / hook 接到 agent

看：

- [Memory System Guide](./docs/memory_system_guide.md)
- [Session Restore Persistence Guide](./docs/session_restore_persistence_guide.md)
- [Callbacks And Streaming Guide](./docs/callbacks_and_streaming_guide.md)
- [Hooks And Guardrails Guide](./docs/hooks_and_guardrails_guide.md)

### 内置工具都有哪些

看：

- [Builtin Tools Catalog](./docs/builtin_tools_catalog.md)

## 11. FAQ

### 文档为什么拆成这么多文件

因为 EasyAgent 不是单一功能库。
Agent、Tool、Prompt、Permission、Runtime、Session、MCP、Memory、Cache 都是独立主题；把所有细节都塞进一个 README 会不可维护。

### README 和 `docs/` 的关系是什么

README 是总入口，负责：

- 快速开始
- 核心概念
- 阅读路径
- 各主题索引

`docs/` 负责：

- 参数逐项解释
- 内部机制说明
- 模块集成细节
- 自定义与扩展方式

### 我应该先看哪几份

通常推荐：

1. `README.md`
2. [Framework API](./docs/framework_api.md)
3. [Agent Guide](./docs/agent_guide.md)
4. [Config Reference](./docs/config_reference.md)
5. 按你的场景继续看 Tool / Prompt / Runtime / Memory / Session
