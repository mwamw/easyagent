# 来源与对比口径

核对日期：2026-09-07。对应正文：`easyagent-whitepaper.zh-CN.md`，版本 v0.1。

## 1. 如何阅读这份比较

- 比较目标是领域 Harness 的构建与定制，不是模型能力、基准分数或市场份额。
- “主要入口”不等于“唯一能力”；项目分类是本文的分析工具，不是互斥的官方类别。
- 项目提供的运行 API、插件、可替换模块以及源码二次开发，属于不同扩展层次。
- 正文不把“公开概览未列出”推断为“项目不支持”，也不以功能打勾数量形成排名。
- 第三方事实使用官方文档或官方仓库。没有使用反编译的 Claude Code 源码作对外结论依据。
- 资料是查阅时的滚动文档快照，不是固定发布版本的全面兼容性测试；发布更新时应重新核对。
- 本版不执行性能、任务成功率、用户学习成本或开发效率实验。

## 2. 官方资料索引

每条概括仅用于说明正文引用范围，属于转述，不是原文摘录。

| 编号 | 官方来源 | 支持的事实与引用边界 |
| --- | --- | --- |
| S01 | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) | 官方区分 LangChain 组件、LangGraph 编排运行时和 Deep Agents Harness。图可包含模型驱动循环，不限于固定步骤。 |
| S02 | [CrewAI introduction](https://docs.crewai.com/en/introduction) | Crews 组织自主协作，Flows 管理状态与流程。核对时重定向到版本化文档；不将其概括为静态 DAG。 |
| S03 | [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents) | 提供 Agent 运行循环、工具、交接、Guardrails 和运行状态；不是只提供跨 Agent 路由。 |
| S04 | [OpenAI Sandbox Agents](https://developers.openai.com/api/docs/guides/agents/sandboxes) | Sandbox Agent 与可组合能力、执行环境接入。正文明确承认其文件、Shell、记忆和压缩能力。 |
| S05 | [Google ADK](https://adk.dev/) | Agent 开发与部署、图式工作流、多模型支持。原官方入口 `google.github.io/adk-docs/` 重定向至此。 |
| S06 | [Microsoft Agent Framework overview](https://learn.microsoft.com/en-us/agent-framework/overview/) | Agent、工作流、状态及企业集成；官方说明与 AutoGen、Semantic Kernel 的继承关系。 |
| S07 | [Hugging Face smolagents](https://huggingface.co/docs/smolagents/index) | 轻量实现、代码行动模型及沙箱选项。CodeAgent 指通过代码表达行动，不等同于只会生成代码的产品。 |
| S08 | [LlamaIndex Workflows introduction](https://developers.llamaindex.ai/python/llamaagents/workflows/) | 事件驱动步骤和应用执行流。旧框架文档路径已重定向；本文没有概括其全部数据平台能力。 |
| S09 | [Haystack Agents](https://docs.haystack.deepset.ai/docs/agents) | Agent 组件自带工具循环，支持状态、流式与人工审核；Pipeline 可作为工具。 |
| S10 | [Codex App Server](https://learn.chatgpt.com/docs/app-server) | 结构化产品集成接口，包括会话、审批及事件。没有据此断言任意内部模块均不能替换。 |
| S11 | [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) | SDK 与 Claude Code 共享工具、循环和上下文能力，提供多种扩展路径。不是“只有终端”的产品。 |
| S12 | [Gemini CLI official repository](https://github.com/google-gemini/gemini-cli) | 开源终端 Agent 及工具/扩展定位。不依赖已失效的 architecture 文档路径作细节判断。 |
| S13 | [OpenCode SDK](https://opencode.ai/docs/sdk/) | 类型化 JS/TS 服务客户端；会话、文件、事件等操作。公开客户端 API 与内部构建接口分别讨论。 |
| S14 | [Aider usage](https://aider.chat/docs/usage.html) | 文件加入对话、代码编辑和 Git 工作方式。未依据该页面陈旧的模型推荐作比较。 |
| S15 | [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview) | 官方定位为可集成的 Harness 库，具有环境、上下文和子 Agent 等能力；并非不可扩展的终端成品。 |
| S16 | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) | Capability 组合、完整 Coder 的可拆解性；承认其与 EasyAgent 的直接竞争关系。 |
| S17 | [OpenHands SDK architecture](https://docs.openhands.dev/sdk/arch/sdk) | Agent、Conversation、Tool、Workspace、Condenser 与 Security 等模块。仅概括其职责，不复述未经源码核对的所有实现细节。 |
| S18 | [Cline SDK architecture](https://github.com/cline/cline/blob/main/sdk/ARCHITECTURE.md) | shared / llms / agents / core 的分层与运行责任，不将模块化或 Core 解耦视为 EasyAgent 独有。 |
| S19 | [OpenHands Software Agent SDK paper](https://arxiv.org/abs/2511.03690) | 已有可组合生产 Agent SDK 的系统论文。仅用于相关工作定位，不挪用其评估结果证明 EasyAgent 的效果。 |

## 3. EasyAgent 的实现依据

核对对象为版本声明 0.8.0 的当前开发工作树，不等同于已发布的发行包。基底提交为 `ac9cb5e`，其上包含未提交的模块重构；只记录提交号无法重建本文描述的完整代码状态。发布白皮书时应与正式标签或归档快照一起发布。

| 主题 | 主要依据 | 可以支持 / 不可以支持 |
| --- | --- | --- |
| 公共入口 | [API 指南](../framework_api.md)、[BasicAgent](../../agent/BasicAgent.py) | 支持默认装配与执行器替换；不证明任意调度细节均有独立插件接口。 |
| Memory | [BaseMemory](../../memory/base.py)、[指南](../memory_system_guide.md) | 说明、工具、Hook 的可选契约和绑定责任；不证明各种文件/外部记忆方案已实现。 |
| Context | [BaseContextManager](../../context/base.py)、[指南](../context_and_compaction_guide.md) | 组装、预算、压缩与来源解耦；不证明完全没有估算误差。 |
| 动态消息 | [MetaMessageEmitter](../../metamessage/emitter.py)、[API 指南](../framework_api.md) | 源无关注入机制；不证明完整事件溯源或任意副作用事务恢复。 |
| Session | [Store 契约](../../core/session/store.py)、[持久化指南](../session_restore_persistence_guide.md) | 显式装配后恢复、存储替换；不是完整 transcript 或外部任务恢复。 |
| Storage | [公共存储指南](../storage_guide.md) | 与领域对象分离的后端和 Embedding 接口；实际部署仍需后端依赖与配置。 |
| 示例 | [模块边界示例](../../example/example_module_boundaries.py) | 公共接口的装配方式；不将示例代码行数用作效率实验。 |
| S4Code | 本工作空间的产品实现与 [模块重构记录](../module_refactor_progress.md) | 产品层/框架层的具体案例；不证明多个领域都已经完成迁移。 |

### S4Code 产品案例

核对了同一工作空间内 S4Code 的 README、`docs/architecture.md` 所描述的产品边界及相关模块重构记录：Core 是主体，`S4CodeAgent` 只是 Agent 组件；CLI、Textual、Python SDK 平级消费 Core；Ink 和 TypeScript SDK 经 Bridge Client 与服务访问 Core。为使此附件随 EasyAgent 仓库分发时不依赖同级仓库路径，正文引用本段作为案例核对记录，不链接工作空间根目录的 AGENTS.md。

S4Code 界面的 checkpoint 只回溯会话上下文，不撤销文件或外部操作。本文的案例仅用于说明框架/产品分工，不提供其效果或稳定性的比较数据。

## 4. 宣传表述的约束

可以使用：“面向 Harness 编程”“机制级记忆契约”“上下文不负责内容检索”“装配与恢复分离”“默认实现与自定义机制共享公共入口”。这些表述必须始终对应具体接口。

需要限定：“降低学习门槛”“减少改动范围”“更方便扩展”“更便于维护”。本文将其作为有机制解释的价值主张，不写成已测量的效果。

不应使用：“全球首个”“唯一真正的 Agent 框架”“任何模块均可无成本替换”“无需任何底层知识”“全面优于所有框架”“生产级安全已被证明”“性能提升若干百分比”。

## 5. 下次更新时重新核对

- 竞品最新公共 API 是否已经覆盖正文涉及的替换需求。
- EasyAgent 发布版本与文中 0.8.0 开发工作树是否一致。
- 记忆、会话和执行环境接口是否发生破坏性变更。
- 链接和产品名称是否重定向或合并。
- S4Code 案例是否仍保持 Core、界面、SDK 的既定责任边界。
