# EasyAgent 🚀

**Build your Agent, not our Agent.**

EasyAgent 是一个面向 **Agent Harness 开发**的 Python Runtime Framework。

它将执行、上下文、记忆、工具、权限、会话与协作组织为可组合的运行能力：提供可直接使用的默认实现，也为关键机制提供可替换的公共契约。你定义领域能力和产品策略，EasyAgent 负责它们如何安装、何时参与默认执行管线，以及如何通过约定协作。

> Framework defines contracts.
>
> Components define capabilities.
>
> Developers define the harness.
>
> Agent is the center.

[快速开始](#quick-start) · [自定义模块](#custom-components) · [架构与能力](#runtime-capabilities) · [框架对比](#comparison) · [文档导航](#documentation) · [技术白皮书](docs/whitepaper/README.md)

Linux 上的内置 `BashTool` 可选使用 Bubblewrap 沙箱。沙箱是 Bash 的执行后端，
不是 Agent Module；支持 regular/auto-allow、严格禁止无沙箱回退、文件系统
读写边界，以及继承网络或完全断网。参见
[Linux Bash Sandbox](docs/bash_sandbox_guide.md)。

## 用语言定义自己的 Harness

先看简短的[配置文件与 DSL 实现概览](docs/configuration_and_dsl.md)（含架构图与对比）。

希望生成可以单独启动的应用目录，可以使用
`buildagent --file agent.yaml --output ./my-agent`，然后运行 `python my-agent/app.py`。
详见[应用生成指南](docs/application_build.md)；支持批量 YAML，且不会覆盖已有目录。

除了 Python 装配和 YAML 配置，现在也提供 [EasyAgent Language](docs/language.md)：
通过 `.ea` 文件声明模型、上下文、工具、权限等能力，支持类型检查、局部导入和参数化模板。
它编译为现有装配定义，运行结果仍是普通 `BasicAgent`，不新增 Agent Loop。

```text
model Main { model = env("EASYAGENT_MODEL") api_key = env("EASYAGENT_API_KEY") }
agent Assistant {
    model = Main
    context { max_tokens = 32000 }
    permission using rule_based { approval_mode = "ask_for_approval" }
    tools { filesystem() }
}
run Assistant with terminal { streaming = true }
```

```bash
easyagent-lang check example/language/minimal.ea     # 无需模型凭据的静态检查
easyagent-lang explain example/language/coding.ea  # 不展示密钥的装配预览
easyagent-lang run example/language/minimal.ea     # 配置模型环境变量后启动
python -m example.language.offline_demo --query "你好"  # 完全离线体验
```

交互界面需安装 `.[ui]`；命令也可以使用 `python -m easyagent.language`。
Python 用于实现新的模块、工具和执行策略，语言用于组合它们；原有 Python/YAML API 不变。

## Why EasyAgent?

专业 Agent 的差异，不只在提示词和工具列表中。

当你构建 Coding、Research、DevOps 或其他长期工具型 Agent 时，可能需要改变：模型每轮看到什么、记忆怎样检索、工具如何执行、什么时候等待审批、会话怎样恢复，以及子 Agent 如何协作。

这些问题属于 **Harness 本身**。EasyAgent 希望让开发者不必在“配置现成 Harness”和“从零实现全部运行设施”之间二选一：

- **直接使用默认能力。** 不必先理解每个模块的内部算法和执行器接线。
- **只实现领域差异。** 用对应模块的接口表达需求，而不是先把所有需求翻译成节点、边或角色任务。
- **由框架承接集成。** 注册、服务绑定和默认执行管线中的调用由公开安装入口衔接。
- **需要时接管运行策略。** 替换执行器，继续复用适用的模型、工具、上下文和状态能力。

EasyAgent 不以另一套工作流 DSL 作为必经入口，也不排斥编排：外层业务流程可以管理任务顺序，内部 Agent 通过 Harness 完成模型驱动的行动。

## Core Philosophy

### Agent is the center

Agent 是能力装配与调用的中心，不意味着所有逻辑都放进一个巨型 Agent 类。工具、记忆、上下文和会话各自承担职责，通过公共接口进入运行过程。

### Framework defines contracts — and provides implementations

框架既不能只有默认实现，也不能只留下一堆需要用户填满的抽象类。

`BasicAgent`、`DefaultAgentExecutor`、`SystemPromptComposer`、`ContextManager`、`LayeredMemory` 和已有存储实现提供起点。自定义模块则使用各职责自己的契约，例如：

`BaseAgentExecutor` · `BaseContextManager` · `BaseHistoryCompactor` · `BaseMemory` · `BaseSystemPromptComposer` · `BaseSessionStore` · `BaseObservabilityManager` · `BaseMultiAgent`

不同能力不必强行继承同一个万能 `AgentModule`。

### Replaceable, not only configurable

配置改变“现有实现怎样工作”；机制替换允许改变“这项能力本身怎样实现”。

例如，换一个向量库和改成模型主动搜索的文件记忆，是两个层次的需求。EasyAgent 将记忆机制与底层存储分开，让两类变化各有位置，而不要求修改默认 Agent Loop。

### Framework-managed integration

**模块作者实现行为，框架承接行为如何进入默认 Agent 运行。**

| 你负责实现 | 框架承接的集成工作 |
| --- | --- |
| Tool 的参数、动作与结果 | 描述导出、参数校验、受控执行、结果与事件 |
| Prompt Composer 的产品规则 | 请求时构建、模块说明合并与请求编译 |
| Memory 的说明、工具与可选 Hook | 服务绑定、工具/Hook 注册及对应生命周期调用 |
| 压缩器的领域保留策略 | 预算预检、调用压缩器、重组候选历史与最终检查 |
| Session Store 的数据操作 | 收集约定状态，保存并恢复到已装配的 Agent |

用户无需理解无关模块的内部实现，也无需分别给同步、异步和流式循环添加同一段业务逻辑。但仍须遵守公开契约：副作用、异步 I/O、资源所有权、状态格式和一致性不是框架能够替用户猜测的内容。

![EasyAgent 的默认实现、模块契约和框架管理的接入](docs/whitepaper/assets/04-framework-architecture.svg)

*功能按需安装，不是全部默认启用。图示说明当前主要职责边界，不表示每个子系统都已具备任意细粒度替换接口。*

## Quick Start

要求 **Python >= 3.10**。从源码安装：

```bash
git clone https://github.com/mwamw/easyagent.git
cd easyagent
python -m pip install -e .
```

按需要安装可选依赖：

```bash
python -m pip install -e ".[mcp]"     # MCP 集成
python -m pip install -e ".[rag]"     # 检索与文档处理
python -m pip install -e ".[memory]"  # 内置记忆方案的可选依赖
python -m pip install -e ".[dev]"     # 开发与测试
```

使用 `BaseMemory` 编写自己的模块，不等于必须安装所有内置记忆依赖。

### 创建第一个 Agent

先设置你实际使用的服务地址、模型和密钥：

```bash
export EASYAGENT_BASE_URL="http://127.0.0.1:5124/v1"
export EASYAGENT_MODEL="your-model"
# 在本地环境中设置 EASYAGENT_API_KEY；不要将真实密钥写入源码。
```

下面是 OpenAI-compatible 协议示例，服务需自行启动或换成可访问的远端地址。这三个环境变量由示例显式读取，不是框架自动识别的统一配置约定。

<!-- readme-example: quick-start -->
```python
import os
from easyagent import BasicAgent, EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url=os.environ["EASYAGENT_BASE_URL"],
    api_key=os.environ["EASYAGENT_API_KEY"],
    model=os.environ["EASYAGENT_MODEL"],
)
agent = BasicAgent(name="assistant", llm=llm)
try:
    print(agent.invoke("用一句话解释什么是 Agent Harness。"))
finally:
    agent.close()
```

最小 `BasicAgent` 始终安装 LLM、Executor、System Prompt 和 Context。Context 拥有历史与请求准备；默认不设总预算上限。Permission、Hook、Callback、Tool、Memory 等附加模块通过 `with_*` 显式安装，不会创建隐式权限策略或 Observability 数据库。

### 显式装配产品能力

下面定义装配函数，接收调用方已经创建的 `llm`，不会在导入时调用模型：

<!-- readme-example: product-assembly -->
```python
from pathlib import Path
from easyagent import BasicAgent, Config
from easyagent.context import ContextManager, LLMHistoryCompactor
from easyagent.tools import ToolRegistry, register_filesystem_tools

def build_product_agent(llm, workspace="."):
    root = Path(workspace).resolve()
    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=str(root))

    context = ContextManager(max_tokens=16_000)
    context.set_history_compactor(
        LLMHistoryCompactor(llm=llm, language="en", recent_turns=4)
    )

    return (
        BasicAgent("product-agent", llm, config=Config(workspace_root=str(root)))
        .with_tool(registry)
        .with_context(context)
        .with_permissions()
        .with_hooks()
        .with_observability(path=str(root / ".easyagent" / "observability.sqlite3"))
    )
```

工具可能读写真实文件；请先配置适合产品的权限和执行环境。LLM 压缩器会在需要时产生额外模型请求。仅设置 `ContextManager(max_tokens=...)` 并不等于安装了摘要算法。

## One Agent, Consistent Invocation APIs

相同 Agent 使用四种调用方式；以下函数接收已装配实例：

<!-- readme-example: invocation -->
```python
def run_sync(agent, query):
    return agent.invoke(query)

async def run_async(agent, query):
    return await agent.ainvoke(query)

def print_stream(agent, query):
    for event in agent.stream(query):
        print(event.to_dict())

async def print_async_stream(agent, query):
    async for event in agent.astream(query):
        print(event.to_dict())
```

是否能调用工具、使用记忆或启动子 Agent，取决于安装的能力；你不需要为每种组合切换另一套 invocation API。流应消费到底，或显式关闭。关闭异步资源时使用 `await agent.aclose()`；共享的模型和存储需要明确所有权，详见 [Agent Guide](docs/agent_guide.md)。

## Custom Components

### 用自己的记忆机制，不改 Agent Loop

`BaseMemory` 表示完整记忆机制，不强制向量检索、固定分类或统一 CRUD。下面的 `tools` 由应用提供，可以是文件搜索、证据读取和保存工具：

<!-- readme-example: custom-memory -->
```python
from easyagent.memory import BaseMemory
from easyagent.prompting import PromptBlock

class ResearchMemory(BaseMemory):
    def __init__(self, retrieval_tools=(), persistence_tools=()):
        self.retrieval_tools = list(retrieval_tools)
        self.persistence_tools = list(persistence_tools)

    def get_prompt_blocks(self, context):
        return [PromptBlock(
            "memory.research",
            "Use the memory tools to search prior findings when relevant. "
            "Save reusable conclusions with their evidence references.",
        )]

    def get_retrieval_tools(self):
        return self.retrieval_tools

    def get_persistence_tools(self):
        return self.persistence_tools

def install_research_memory(agent, retrieval_tools=(), persistence_tools=()):
    return agent.with_memory(ResearchMemory(retrieval_tools, persistence_tools))
```

安装时，框架绑定服务并注册模块提供的工具和 Hook；请求准备时，正常的 Prompt Composer 合并记忆说明。模型调用记忆工具时，执行和结果走普通工具路径。

不需要自动检索，就不提供检索 Hook；需要自动检索，可以由 Memory 自己提供公开 `before_context` / `abefore_context` Hook，再通过 MetaMessage 注入。**Context 不替 Memory 决定检索机制。**

内置 `LayeredMemory` 是另一种实现。Working、Episodic、Semantic、Perceptual 等分类属于该分层方案，不是 `BaseMemory` 对所有自定义系统的要求。

### 自定义提示词，不重写请求编译

通常只需实现 `build()`。基类的 `compose()` 在启用模块说明时继续合并 Memory 的 Prompt Blocks：

<!-- readme-example: custom-prompt -->
```python
from easyagent.prompting import BaseSystemPromptComposer, PromptBlock

class ResearchPrompt(BaseSystemPromptComposer):
    def build(self, context):
        return [PromptBlock(
            "identity",
            "You are a research assistant. Separate evidence from hypotheses.",
        )]

def install_research_prompt(agent):
    return agent.with_prompt(ResearchPrompt())
```

也可以直接配置 `SystemPromptComposer(..., include_defaults=False)`，关闭框架默认规则。模块说明是否合并是另一个选项，不应因改变产品身份而意外丢失。

### 关键扩展入口

| 需要改变的能力 | 公共契约 / 对象 | 安装方式 |
| --- | --- | --- |
| Agent Loop | `BaseAgentExecutor` | `with_executor(executor)` |
| 上下文组装 | `BaseContextManager` | `with_context(manager)` |
| 历史压缩 | `BaseHistoryCompactor` | `context.set_history_compactor(compactor)` |
| 记忆机制 | `BaseMemory` | `with_memory(memory)` |
| 产品提示词 | `BaseSystemPromptComposer` | `with_prompt(composer)` |
| 工具 | `Tool` / `ToolRegistry` | `add_tool(tool)` / `with_tool(registry)` |
| 权限策略 | `BasePermissionEngine` | `with_permissions(engine)` |
| 会话存储 | `BaseSessionStore` | `with_session(store)` |
| 可观测性 | `BaseObservabilityManager` / `BaseObservabilityStore` | `with_observability(manager=...)` 或 `store=...` |
| Multi-Agent | `BaseMultiAgent` | `with_multi_agent(module)` |

`with_skill`、`with_mcp`、`with_plan`、`with_task_service`、`with_hooks`、`with_callbacks`、`with_codeintel`、`with_worktree`、`with_interruptions` 提供其他能力入口。

整体替换执行器意味着接管请求、工具、权限、事件和状态之间的协调责任；框架提供可复用对象，但不会自动推断你的新循环。局部需求优先使用局部接口，不必复制整个 Executor。

更多可验证示例见 [模块边界示例](example/example_module_boundaries.py) 与 [白皮书第 3、4 章](docs/whitepaper/easyagent-whitepaper.zh-CN.md)。

## Runtime Capabilities

### History、Context、Memory、Compaction 各有职责

| 概念 | 回答的问题 | 当前边界 |
| --- | --- | --- |
| History | 当前保留了哪些对话与动作信息？ | 内存中维护 canonical 与 provider replay；不是永不丢失的原始日志 |
| Context | 这一轮模型看到什么？ | 组装准备好的内容，控制总预算，不主动检索 Memory / RAG |
| Memory | 怎样保存并使用可复用知识？ | 通过说明、工具和可选 Hook 参与运行，机制由实现决定 |
| Compaction | 输入过长时怎样缩减历史？ | 规则式、LLM 或自定义压缩器；压缩可能替换当前历史 |
| MetaMessage | 动态信息怎样注入与回收？ | 处理消息与生命周期，不理解来源的业务含义 |

Canonical History 与 Provider Replay 分离，使产品不必把持久化状态直接绑定到某一种 provider 消息格式。跨模型恢复仍可能存在能力差异，需要查看恢复报告。

框架预算判断优先采用**有效 API usage 基线 + 本地增量估算**；首轮或基线失效时回退完整本地估算。压缩预检、最终请求检查和 `get_context_usage()` 遵循同一规则，不依赖启用 Observability。

### Tool：不仅是一个函数

`Tool`、`ToolSpec`、`ToolResult` 与 `ToolRegistry` 一起表达参数 Schema、读写语义、副作用、权限、可见性、按需展开和执行结果。普通工具、内置工具与 MCP 工具沿相关运行路径参与 Agent。

Deferred 模式使用 `tool_schema_tool` 按需展开完整 Schema，减少初始请求携带的完整工具定义。注意当前字段语义：**`expose_in_deferred=True` 表示在 deferred 模式下直接暴露**；设为 `False` 的工具需按需展开。工具 Schema 标记不等同于所有执行器都会并行调度工具。

### Permission、Hook 与 Callback 分离

- **Permission**：对动作作 Allow / Ask / Deny 决策。
- **Hook**：在公开生命周期位置检查、修改或阻断。
- **Callback / Observability**：观察执行事实，生成展示、日志或记录。

当前默认工具路径先运行 `before_tool_use`，再校验修改后的工具调用并授权，之后执行工具、处理结果与发布相应事件。因此 Hook 改写参数后仍需经过权限判断。

**Permission、路径限制、Worktree 都不等于 OS Sandbox。** 文件、Shell、网络和进程操作应结合合适的容器或操作系统隔离。审批等待需要产品提供用户交互入口，不能只靠提示词保证安全。

### Skill、MCP 与动态能力

目录式 Skill 使用 `SKILL.md`。安装时索引可用技能，模型按需加载正文；框架处理 listing、激活与临时信息回收。它不要求用户实现额外的 Python 模块接线。

MCP 将外部工具和资源接入本地工具体系，并处理相应连接、Schema、认证与策略边界。远程 prompt 与本地 Skill 不会被无条件合并，产品需明确决定如何使用远端内容。

详见 [Skill](docs/skill_system_guide.md)、[MCP](docs/mcp_guide.md)、[Deferred Tools](docs/deferred_tools_guide.md)。

### Subagent、Team、Task 与 Mailbox

可选 `ClaudeCodeMultiAgent` 是内置的 Claude Code 风格实现，提供 Agent Handle、后台运行、查询/等待/停止、Team、自动消息投递和 Completion Record。框架层只定义 `BaseMultiAgent` 的工具、提示词、上下文前置处理、状态和关闭接口，因此也可以换成完全不同的 Actor、Swarm 或远程 Worker 实现。

Mailbox 投递与模型读取分开：框架在后续请求准备时同步消息，通过 MetaMessage 接入上下文并去重；读取和确认仍有独立操作。持久化 Handle 不意味着外部进程会随会话自动复活。

### Environment 与 Code Agent 能力

EasyAgent 提供文件、Shell、CodeIntel、Worktree 等能力，供产品构建操作真实环境的 Agent。它不是只面向 Coding Agent；Browser、Computer-use 或远端计算等场景，还需要相应工具和环境实现，不能理解为已经内置所有专用 Harness。

在前面的装配方式上，Coding Agent 可以这样组织工具：

<!-- readme-example: code-agent -->
```python
from pathlib import Path
from easyagent import BasicAgent, Config
from easyagent.tools import (
    ToolRegistry, register_filesystem_tools, register_shell_tools,
)

def build_code_agent(llm, workspace="."):
    root = Path(workspace).resolve()
    registry = ToolRegistry()
    register_filesystem_tools(registry, workspace_root=str(root))
    register_shell_tools(registry, workspace_root=str(root), expose_in_deferred=False)
    return BasicAgent(
        "code-agent", llm,
        config=Config(workspace_root=str(root), tool_schema_mode="deferred"),
    ).with_tool(registry).with_permissions().with_hooks()
```

上面仅装配能力，不执行命令。多个 Agent 修改同一仓库时，可用独立 Worktree 分开工作副本；Git 分支合并、外部资源和执行隔离仍需产品管理。

## Session & Recovery

会话持久化保存**当前可恢复状态**，不根据实现类名猜测产品装配。先创建所需模型、工具和模块，再恢复数据：

<!-- readme-example: session -->
```python
from easyagent import BasicAgent
from easyagent.session import InMemorySessionStore

def demonstrate_session(llm):
    store = InMemorySessionStore()  # 仅本进程；跨进程保存可换 SQLiteSessionStore
    original = BasicAgent("assistant", llm).with_session(store)
    restored = BasicAgent("assistant", llm).with_session(store)
    try:
        original.add_user_message("Remember the current research objective.")
        session_id = original.save_session()
        restored.restore_session(session_id)
        branch_id = restored.fork_session()
        return session_id, branch_id, restored.get_last_restore_report()
    finally:
        original.close(close_llm=False)
        restored.close(close_llm=False)
```

共享 `llm` 由调用方关闭。真实产品应通过自己的装配函数创建恢复实例；自定义模块状态通过约定的导出/恢复接口与状态参与者机制保存。

恢复可能遇到 provider 变化、缺失工具或缺失模块。`SessionRestoreReport`、`ComponentRestoreReport` 和 `RestoreIssue` 让产品看清可恢复内容与差异；必需组件缺失或状态无效时会报错，而不是继续假装恢复成功。

**保存/恢复不自动回滚文件、重新连接所有外部资源或恢复执行中的进程。** 当前快照也不是完整 transcript archive。旧数据库应显式迁移，详见 [Session Guide](docs/session_restore_persistence_guide.md) 和 [Storage Guide](docs/storage_guide.md)。

## Events, Observability & Training

`RuntimeEventBus` 提供统一执行事实，如 `agent.invoke.started`、`llm.invoke.completed`、`tool.invoke.completed`、`history.compacted`。流接口使用 `text_delta`、`reasoning_delta`、`tool_call`、`tool_result`、`final`、`error` 等结构化事件。

Observability 按需订阅事件，记录 `AgentInvoke`、`LLMInvoke`、工具调用、usage、耗时、异常和输入输出，支持内存、SQLite 及自定义 Store。TrainingExporter 消费这些记录，不要求 Executor 理解训练格式：

```text
Agent / Tools → Runtime Events → Observability → Training Exporter
                        └──────→ Callback / UI / Logs
```

当前事件总线是同步进程内分发。观察者不应执行耗时阻塞工作；“观察与执行分离”不是“所有回调天然非阻塞”。观测记录可能含提示词、工具结果及敏感数据，落库和导出前应控制访问并按需脱敏。

## Provider-neutral, Not Provider-blind

`EasyLLM` 提供统一模型入口，支持 `openai`、`openai_responses`、`anthropic_native`、`google_native` 等协议路径。

统一请求、历史转换、工具调用和 usage，不意味着所有 provider 的 reasoning、缓存或消息语义完全相同。产品可以使用受支持的特定能力，也需要核对模型与 provider 的限制。框架内部的 provider 实现不应变成每个产品都要维护一套的 EasyAgent 适配层。

## Comparison

不同项目的抽象可以重叠。下表比较主要公开入口，不作性能或成熟度排名；项目链接指向官方说明，核对日期为 2026-09-10。

| 框架 / 路线 | 主要抽象与能力 | 更直接服务的需求 |
| --- | --- | --- |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | State / Node / Edge；持久执行、流式与人工介入 | 组织有状态流程，可混合确定性与模型驱动步骤 |
| [CrewAI](https://docs.crewai.com/en/introduction) | Agent / Role / Task / Crew / Flow | 角色协作与业务执行流程 |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/)（AutoGen / Semantic Kernel 后续方向） | Agent、工作流、会话、中间件与遥测 | 多 Agent 应用和企业集成 |
| [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents) | Agent / Runner / Tool / Handoff，以及运行状态等能力 | 以代码定义、运行和集成 Agent 应用 |
| [Deep Agents](https://docs.langchain.com/oss/python/deepagents/customization) | 已装配的 Harness、中间件与可接入后端 | 使用并定制完整 Harness，包括替换默认中间件 |
| [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) | Capability 组合与可拆解的 Agent 能力 | 通过组合能力构建和定制 Harness |
| [OpenHands SDK](https://docs.openhands.dev/sdk/arch/sdk) | Agent / Conversation / Workspace / Tools / Condenser | 软件 Agent 及可复用的运行与环境设施 |
| **EasyAgent** | **Agent Harness 的职责契约与可组合实现** | **直接定制执行、记忆、上下文与会话等领域机制** |

EasyAgent 的取向是 **Use our framework to build your harness**：将关键运行职责作为公共接口，提供默认实现，并将产品策略与框架接入机制分开。模块化不是 EasyAgent 独有；真正应比较的是某个需求能通过什么接口完成、需要理解哪些概念，以及替换后由谁承担协作责任。

更完整的 17 组项目对照、十二个维度和扩展示例见 [技术白皮书](docs/whitepaper/README.md)（其资料核对日期单独标注）。

## Grow Without Rebuilding Everything

从最小 Agent 到领域产品，能力可以逐步增加：

1. 使用默认 Agent 和模型入口验证需求。
2. 加入 Tools、Context、权限与所需的状态管理。
3. 接入 Memory、Skill、MCP、Task 或 Subagent。
4. 按领域需要替换提示词、记忆、压缩器和会话存储。
5. 循环策略真正不同时，再提供自定义 Executor。

这条路线旨在减少因局部需求而 fork 整个框架的必要性，而不是承诺永远无需修改框架。当前固定的历史组件、执行环境和工具调度等仍有边界；例如，完全无消息历史的事件溯源 Agent、任意环境后端无缝替换，不应宣传成现成能力。

## When to Use EasyAgent

适合希望掌握 Agent 运行机制的团队：构建领域工具型产品，尝试不同记忆架构、上下文策略、Agent Loop 或协作方式，并让 CLI、TUI、服务与产品 SDK 共享自己的 Core。[S4Code](https://github.com/mwamw/S4Code) 是一个基于 EasyAgent 的产品案例，不是框架限定的产品形态。

如果只是一次模型调用或简单检索，模型 SDK 或现成数据工具可能更直接；如果核心需求是确定性业务流程和持久工作流，工作流运行时可能更合适；如果完整 Harness 已满足要求，直接用其 SDK 也很合理。

## Documentation

无需编写装配代码时，可以使用新增的[配置文件加载入口](docs/configuration_loading.md)：
`load_agent("agent.yaml")` 返回普通 Agent，原有 Python API 不变；
`easyagent-run agent.yaml --check` 可先进行静态校验。

安装可选依赖 `pip install -e ".[ui]"` 后，可使用
`easyagent-chat agent.yaml` 启动[默认终端 UI](docs/terminal_ui.md)，支持流式对话、
工具状态、权限确认与用户提问。原有 Python API 和单请求入口保持可用。

| 主题 | 指南 |
| --- | --- |
| 入门与架构 | [Framework API](docs/framework_api.md) · [Agent Guide](docs/agent_guide.md) · [模块化架构](docs/modular_agent_architecture.md) · [Config](docs/config_reference.md) · [Provider](docs/llm_provider_guide.md) |
| Harness 设计 | [技术白皮书及 PDF](docs/whitepaper/README.md) · [可执行装配示例](example/example_module_boundaries.py) · [Code Agent Quickstart](docs/code_agent_product_quickstart.md) |
| 上下文与记忆 | [Context & Compaction](docs/context_and_compaction_guide.md) · [Memory](docs/memory_system_guide.md) · [Storage](docs/storage_guide.md) · [RAG](docs/rag_guide.md) |
| 提示词与技能 | [Prompt System](docs/prompt_system_guide.md) · [Prompt Composer](docs/prompt_composer_guide.md) · [Skill](docs/skill_system_guide.md) |
| 工具与安全 | [Tool System](docs/tool_system_guide.md) · [Tool Authoring](docs/tool_authoring_guide.md) · [Builtin Tools](docs/builtin_tools_catalog.md) · [Deferred Tools](docs/deferred_tools_guide.md) · [Permission](docs/permissions_guide.md) · [Hooks](docs/hooks_and_guardrails_guide.md) |
| 状态与协作 | [Session](docs/session_restore_persistence_guide.md) · [Runtime / Mailbox / Subagent](docs/runtime_collaboration_guide.md) · [Tasks](docs/tasks_guide.md) |
| 扩展与观测 | [MCP](docs/mcp_guide.md) · [CodeIntel](docs/codeintel_guide.md) · [Worktree](docs/worktree_guide.md) · [Callbacks / Streaming](docs/callbacks_and_streaming_guide.md) · [Observability / Training](docs/observability_and_cache_guide.md) |

新项目从 `easyagent` 或 `easyagent.*` 导入。内部的 `agent/`、`core/`、`Tool/`、`runtime/` 等目录不是产品侧稳定导入入口。

## Development & Project Status

在已配置的 Python 环境中运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest test -q
```

默认测试不要求付费模型调用；集成与外部服务测试需要显式启用，并自行准备依赖及凭据。

当前版本为 **0.8.0**，仍在持续演进。项目重点是打磨执行、上下文、记忆、工具、权限、会话和协作的边界，而不是无限增加功能。生产集成请锁定已验证的版本，关注接口变更、迁移和所使用模块的能力范围。版本声明和本地可安装不等同于已发布到包仓库。

## Vision

EasyAgent 希望封装的，不只是重复代码，还有模块如何参与 Agent 运行的系统知识。

**让使用默认能力的人不必成为 Harness 专家，让需要改变机制的人不必重建整个 Harness。**

Build your Agent, not our Agent.

## License

MIT.
