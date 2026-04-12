# EasyAgent 待实现功能 (Roadmap)

本文档记录 EasyAgent 框架后续计划实现的功能。
> 最后更新：2026-04-12

---

## 🔴 高优先级 (P1)

### 1. 原生异步支持 (已完成)
当前异步只是 `run_in_executor` 包装，需要原生 `async/await`。

- `AsyncOpenAI` 客户端支持
- Provider 层 `async invoke` / `async stream`
- Agent 层完整异步生命周期

```python
async def main():
    response = await agent.ainvoke(query)
    async for chunk in agent.astream(query):
        print(chunk)
```

**涉及文件:**
- `core/providers/base.py` - 添加 `async_invoke`, `async_stream` 等
- `core/llm.py` - 添加 `ainvoke`, `astream`
- `agent/*.py` - 异步版本

---

### 2. 对话持久化与会话管理 (部分完成 后续完善)
对话历史和 Agent 状态目前只存在内存中，需要支持持久化。

- SQLite / Redis 对话存储
- 会话管理（创建、恢复、列表、过期清理）
- Agent 状态快照与恢复

```python
agent.save_session("session_001")
agent = BasicAgent.load_session("session_001")
```

**涉及文件:**
- `db/session_store.py` - 会话存储
- `db/conversation_store.py` - 对话持久化
- `core/agent.py` - Agent 状态序列化

**已完成功能:**
- SQLite 对话存储
- 会话管理（创建、恢复、列表、过期清理）
- Agent 状态快照与恢复

**后续完善：**
- Redis 对话存储

---

### 3. 更多向量/图数据库支持
当前仅支持 Qdrant 向量存储和 Neo4j 图存储，扩展更多后端。

- Milvus VectorStore 实现
- Pinecone VectorStore 实现
- `memory/V2/Store/MilvusVectorStore.py`
- `memory/V2/Store/PineconeVectorStore.py`

---

### 4. 多 Agent 协作增强
当前已实现 Sequential / Supervisor / GroupChat 三种编排模式，后续增强：

- **DAG 工作流编排**：支持并行/串行/条件分支的复杂工作流
- **共享记忆**：多个 Agent 通过 SharedContext 访问同一 V2 记忆系统
- **Agent 间异步通信**：支持跨 Agent 异步消息传递
- **动态 Agent 创建与销毁**：运行时按需创建/销毁子 Agent

**涉及文件:**
- `orchestrator/dag.py` - DAG 工作流编排器
- `orchestrator/context.py` - 增强 SharedContext 与 V2 Memory 集成

---

## 🟡 中优先级 (P2)

### 5. Guard Rails / 安全机制
- 输入检查（Prompt Injection 防护）
- 输出过滤（敏感内容检测）
- 工具调用权限控制（白名单/黑名单）
- Token 用量限制 / 预算控制

**涉及文件:**
- `core/guardrails.py` - 安全检查框架
- `core/token_budget.py` - Token 预算管理

---

### Tool 模块后续完善
当前 `Tool` 模块已经完成 `ToolSpec + ToolResult + ToolRegistry` 的主干重构，并与 `Skill` 的 on-demand 临时工具挂载打通；后续重点是把协议层进一步工程化。

- 更完整的工具权限/风险语义：
  - 除 `requires_confirmation` 之外，增加更细粒度的 destructive / network / filesystem / side-effect 分级
  - 为高风险工具建立统一确认与拒绝反馈路径
- 更完整的 provider-neutral schema 适配：
  - 不再只以 OpenAI 风格 function schema 为事实标准
  - 为 Anthropic / Google / Responses API 统一输出适配层
- 更严格的工具可见性与冲突管理：
  - 对同名工具覆盖策略做显式约束
  - 强化 resident / runtime / turn 工具的生命周期管理
  - 避免临时工具与常驻工具发生静默冲突
- 更体系化的 runtime context：
  - 继续保留 `ToolResult.ephemeral_context`
  - 让工具临时上下文在 compaction / session rebuild / trace 中也有稳定协议

**涉及文件:**
- `Tool/BaseTool.py`
- `Tool/ToolRegistry.py`
- `core/agent.py`
- `agent/BasicAgent.py`

---

### MCP 模块增强
当前 EasyAgent 的 MCP 集成已经能把远程 MCP tools 包装成 EasyAgent Tool，但相对 Claude Code 仍然偏轻，后续需要补全 MCP 生态层能力。

- MCP resources 支持：
  - 增加资源列举与读取能力，对齐 `ListMcpResources` / `ReadMcpResource`
  - 区分“远程 tools”和“远程 resources”两类能力
- MCP prompts / commands 支持：
  - 支持从 MCP server 拉取 prompt/command，并映射到 EasyAgent 的 Skill / Command / Prompt 体系
  - 避免 MCP 只剩“远程工具桥接”这一条窄路径
- MCP 权限与连接管理增强：
  - 更细粒度的 server/tool 级权限控制
  - 自动重连、错误分类、结果缓存与失效策略
  - 与 ToolResult / ToolSpec 的风险语义打通
- MCP 能力发现与展示优化：
  - 区分本地 builtin tool、Skill 临时 tool、MCP remote tool
  - 给模型提供更稳定的远程能力边界说明

**涉及文件:**
- `Tool/builtin/mcp_tool.py`
- `mcp/`
- `skill/meta_tools.py`
- `prompt/system_prompt.py`

---

### 6. 流式 + 工具调用组合 (已完成)
已支持「边流式输出边触发工具」的 Streaming Tool Call 模式，`BasicAgent` 的同步/异步流式入口都可在工具模式下工作。

**涉及文件:**
- `agent/BasicAgent.py` - `stream_invoke_with_tool`
- `core/providers/base.py` - `stream_with_tools`
- `core/providers/openai_responses_provider.py` - Responses API 流式工具事件解析
- `core/llm.py` - 统一流式工具调用入口

**已完成功能:**
- `BasicAgent.stream_invoke()` / `astream_invoke()` 支持工具模式
- OpenAI-compatible Provider 流式工具调用
- OpenAI Responses Provider 流式工具调用
- Anthropic / Google Provider 消息格式适配
- 统一事件流：`thinking_delta` / `text_delta` / `tool_call` / `tool_result` / `final`

**说明文档:**
- `docs/streaming.md`
- `docs/history_and_messages.md`

---

### 7. 可观测性 (Observability)
- OpenTelemetry Tracing 集成
- 每次调用的 Token 计数（input/output）
- 调用链可视化（Agent → LLM → Tool → LLM → ...）
- 成本自动计算

**涉及文件:**
- `core/tracing.py` - Tracing 框架
- `core/callbacks.py` - 扩展 `MetricsCallback` 支持 Token 计数

---

### 8. 更多 Agent 模式
- **ReWOO Agent**：先规划所有工具调用，再批量执行
- **Reflexion Agent**：自反思 + 自我修正
- **Code Interpreter Agent**：动态执行代码

**涉及文件:**
- `agent/ReWOOAgent.py`
- `agent/ReflexionAgent.py`
- `agent/CodeInterpreterAgent.py`

---

### 9. 更多预置工具 / 内置 Skill
当前内置 Skill 有 calculator、web_search、memory、mcp，继续扩展：

- `skill/builtin/code_interpreter_skill.py` - 代码执行器 Skill
- `skill/builtin/file_manager_skill.py` - 文件管理 Skill
- `skill/builtin/http_client_skill.py` - HTTP 请求 Skill
- `skill/builtin/linux_ops_skill.py` - Linux 运维 Skill

---

### 10. V2 记忆性能优化
- 批量 embedding 管道优化
- 向量缓存层减少重复编码
- 异步并发存储写入

---

### 11. Skill 远程仓库支持
当前 Skill 仅支持本地加载（YAML / Markdown / Folder），增加远程加载能力：

- Skill 远程仓库协议定义
- 从 Git / HTTP 加载 Skill
- Skill 版本管理与依赖声明

**涉及文件:**
- `skill/remote_loader.py` - 远程 Skill 加载器
- `skill/registry.py` - 增加远程注册源

---

## 🟢 低优先级 (P3)

### 12. Agent 评估系统
- Agent 质量评估框架（准确性、幻觉率、工具调用正确率）
- Benchmark 数据集支持
- A/B 对比不同 Agent 配置

---

### 13. 工程化打包
- `pyproject.toml` + `pip install easyagent`
- CI/CD（GitHub Actions / pre-commit hooks）
- 自动文档生成（mkdocs / sphinx）
- CHANGELOG 版本管理

---

### 14. Web UI 管理界面
基于 Gradio/Streamlit 的可视化管理界面，支持对话调试、记忆查看、Agent 配置。

---

### 15. 多模态支持
- Vision（图片输入 / 图片理解）
- Audio（语音输入/输出）
- 视频模态为 PerceptualMemory 添加视频编码检索

---

### 16. 分布式 Agent
支持在多台机器上运行 Agent 集群，任务自动分发与负载均衡。

---

### 17. 更多 LLM 提供商支持
当前已支持 OpenAI / Anthropic / Google 三大主流 Provider，后续扩展：

- 讯飞星火
- 百川
- MiniMax

---

## ✅ 已完成

| 功能 | 完成日期 | 版本 |
|------|----------|------|
| **Skill 技能系统** | 2026-04 | v2.2-dev |
| 流式输出 + 工具调用组合 | 2026-04 | v2.2-dev |
| SQLite 会话持久化与恢复 | 2026-04 | v2.2-dev |
| Skill 基础架构 (BaseSkill, SkillConfig, SkillManager) | 2026-04 | v2.2-dev |
| SkillRegistry 注册中心（关键词/标签搜索） | 2026-04 | v2.2-dev |
| YAMLSkill / MarkdownSkill 声明式加载 | 2026-04 | v2.2-dev |
| FolderSkill 文件夹加载器 | 2026-04 | v2.2-dev |
| 动态 Skill 加载 — 模式 B (MetaSkill) | 2026-04 | v2.2-dev |
| Meta-Tools (SkillDiscoveryTool / LoadSkillTool / UnloadSkillTool) | 2026-04 | v2.2-dev |
| 内置 Skill (calculator / web_search / memory / mcp) | 2026-04 | v2.2-dev |
| Agent 集成 SkillManager (BasicAgent / ReactAgent / PlanningAgent / StructuredOutputAgent) | 2026-04 | v2.2-dev |
| Skill 单元测试 | 2026-04 | v2.2-dev |
| **多 Agent 协作 (Multi-Agent Orchestration)** | 2026-03 | v2.1-dev |
| SequentialOrchestrator 顺序编排 | 2026-03 | v2.1-dev |
| SupervisorOrchestrator 主管模式 | 2026-03 | v2.1-dev |
| GroupChatOrchestrator 群聊模式 | 2026-03 | v2.1-dev |
| AgentMessage / SharedContext / 异常体系 | 2026-03 | v2.1-dev |
| Orchestrator 回调系统集成 | 2026-03 | v2.1-dev |
| Orchestrator 单元测试 | 2026-03 | v2.1-dev |
| **Context Engineering 上下文工程模块** | 2026-03 | v2.1-dev |
| ContextBuilder / ContextManager / ContextWindow | 2026-03 | v2.1-dev |
| 多源上下文收集 (HistorySource / RAGSource / MemorySource) | 2026-03 | v2.1-dev |
| Token 预算管理 (TokenCounter / TokenBudget) | 2026-03 | v2.1-dev |
| 压缩器 (SlidingWindow / TokenBudget / Selective / Summarization) | 2026-03 | v2.1-dev |
| 格式化器 (Plain / XML / Markdown) | 2026-03 | v2.1-dev |
| **RAG 检索增强生成模块** | 2026-03 | v2.1-dev |
| DocumentLoader 文档加载器 (30+ 格式) | 2026-03 | v2.1-dev |
| 分块策略 (Fixed / Recursive / Semantic / Token) | 2026-03 | v2.1-dev |
| 嵌入模型 (OpenAI / HuggingFace) | 2026-03 | v2.1-dev |
| 向量存储 (Memory / ChromaDB) | 2026-03 | v2.1-dev |
| 检索器 (Vector / BM25 / Hybrid / MultiQuery / ReRank / Compression) | 2026-03 | v2.1-dev |
| 查询转换 (HyDE / Step-Back) | 2026-03 | v2.1-dev |
| RAGPipeline 管线编排 | 2026-03 | v2.1-dev |
| **MCP 模块增强 (MCPServer)** | 2026-03 | v2.1-dev |
| MCP Server 端实现 (mcp_server.py) | 2026-03 | v2.1-dev |
| **P0/P1 Bug 修复与代码质量提升** | 2026-03-21 | v2.1-dev |
| temperature=0.0 bug 修复 | 2026-03-21 | v2.1-dev |
| 命名 typo 修复 (resovle→resolve, provide→provider_name) | 2026-03-21 | v2.1-dev |
| 回调系统集成到 Agent (on_agent/tool/llm_start/end) | 2026-03-21 | v2.1-dev |
| 统一 snake_case 命名 (ToolRegistry + BaseAgent) | 2026-03-21 | v2.1-dev |
| Provider 代码去重 (提取通用逻辑到 BaseProvider) | 2026-03-21 | v2.1-dev |
| **V2 记忆系统与 Agent 集成** | 2026-03 | v2.0-dev |
| V2 感知记忆 (PerceptualMemory) | 2026-03 | v2.0-dev |
| V2 感知记忆 `load_from_store` / `sync_stores` | 2026-03 | v2.0-dev |
| **V2 语义记忆 (SemanticMemory)** | 2026-02 ~ 2026-03 | v2.0-dev |
| 语义记忆向量+图谱混合排序 | 2026-03 | v2.0-dev |
| **Neo4j 图存储 (Neo4jGraphStore)** | 2026-02 ~ 2026-03 | v2.0-dev |
| **LLM 实体关系提取器 (Extractor)** | 2026-02 | v2.0-dev |
| **V2 情景记忆 (EpisodicMemory)** | 2026-02 | v2.0-dev |
| 情景记忆批量/异步支持 | 2026-02 | v2.0-dev |
| 情景记忆模式发现 (find_patterns) | 2026-02 | v2.0-dev |
| **V2 工作记忆 (WorkingMemory)** | 2026-02 | v2.0-dev |
| **V2 基础架构** (BaseMemory, MemoryConfig, Stores, Embedding) | 2026-02 | v2.0-dev |
| SQLite 文档存储 | 2026-02 | v2.0-dev |
| Qdrant 向量存储 | 2026-02 | v2.0-dev |
| HuggingFace 嵌入模型 | 2026-02 | v2.0-dev |
| 异步工具执行器 (AsyncToolExecutor) | 2026-02 | v2.0-dev |
| **MCP 支持 (MCPClient + MCPToolWrapper)** | 2026-02 | v2.0-dev |
| Callbacks 回调系统 | 2026-01-19 | v1.0 |
| **Provider 适配器模式** | 2026-01-19 | v1.1 |
| Anthropic Provider 适配器 | 2026-03 | v2.1-dev |
| Google Provider 适配器 | 2026-03 | v2.1-dev |
| OpenAI Responses Provider 适配器 | 2026-03 | v2.1-dev |
| ConversationSummaryMemory | 2026-01-19 | v1.0 |
| StructuredOutputAgent | 2026-01-19 | v1.0 |
| WebSearchTool | 2026-01-19 | v1.0 |
| CalculatorTool | 2026-01-19 | v1.0 |
| 单元测试 (52+ tests) | 2026-01-19 | v1.0 |

---

## 贡献指南

如果你想贡献代码，请：
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交代码 (`git commit -m 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request
