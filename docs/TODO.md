# EasyAgent 待实现功能 (Roadmap)

本文档记录 EasyAgent 框架后续计划实现的功能。

---

## 🔴 高优先级 (P1)

### 1. 多 Agent 协作 (Multi-Agent Orchestration)
支持多个 Agent 协作完成复杂任务，是框架从"单体 Agent"演进为"Agent 平台"的核心能力。

- **Supervisor 模式**：一个主 Agent 调度多个子 Agent
- **Agent 间消息传递**：标准化的通信协议
- **工作流编排**：DAG 式 Agent Pipeline，支持并行/串行/条件分支
- **共享记忆**：多个 Agent 访问同一记忆系统

```python
# 预期用法
orchestrator = AgentOrchestrator(
    supervisor=supervisor_agent,
    workers={
        "researcher": researcher_agent,
        "writer": writer_agent,
        "reviewer": reviewer_agent,
    },
    workflow="researcher -> writer -> reviewer"
)
result = orchestrator.run("写一篇关于 AI Agent 的调研报告")
```

**涉及文件:**
- `agent/orchestrator.py` - 编排引擎
- `agent/communication.py` - Agent 间通信
- `core/workflow.py` - 工作流定义与执行

---

### 2. Skill 技能系统
允许 Agent 加载和使用预定义的"技能包"，每个 Skill 封装一组工具 + 提示词 + 工作流。

- **Skill 定义**：YAML/Python 声明式技能描述
- **Skill 注册与发现**：从本地目录或远程仓库加载
- **Skill 组合**：多个 Skill 可以叠加到同一个 Agent
- **内置 Skill**：代码执行、文件操作、网络搜索、数据分析等

```python
# 预期用法
from skill import SkillManager, Skill

skill_mgr = SkillManager()
skill_mgr.load_from_dir("./skills/")  # 加载本地技能包

# 单个 Skill 结构
# skills/code_review/
# ├── skill.yaml          # 技能元数据 (name, description, tools, prompts)
# ├── tools.py            # 技能专属工具
# └── prompt_templates/   # 技能提示词模板

agent = BasicAgent(name="dev", llm=llm, skill_manager=skill_mgr)
agent.use_skill("code_review")
agent.use_skill("web_search")
```

**涉及文件:**
- `skill/base.py` - Skill 基类定义
- `skill/manager.py` - SkillManager 管理器
- `skill/loader.py` - Skill 加载器（本地/远程）
- `skill/builtin/` - 内置技能包
- `core/agent.py` - BaseAgent 集成 SkillManager

---

### 3. 原生异步支持
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

### 4. 对话持久化与会话管理
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

---

### 5. 更多向量/图数据库支持
- Milvus VectorStore 实现
- Pinecone VectorStore 实现
- `memory/V2/Store/MilvusVectorStore.py`
- `memory/V2/Store/PineconeVectorStore.py`

---

## 🟡 中优先级 (P2)

### 6. Guard Rails / 安全机制
- 输入检查（Prompt Injection 防护）
- 输出过滤（敏感内容检测）
- 工具调用权限控制（白名单/黑名单）
- Token 用量限制 / 预算控制

**涉及文件:**
- `core/guardrails.py` - 安全检查框架
- `core/token_budget.py` - Token 预算管理

---

### 7. 流式 + 工具调用组合
当前 `stream_invoke` 不支持工具模式。需要支持「边流式输出边触发工具」的 Streaming Tool Call 模式。

**涉及文件:**
- `agent/BasicAgent.py` - `stream_invoke_with_tool`
- `core/providers/base.py` - `stream_with_tools`

---

### 8. 可观测性 (Observability)
- OpenTelemetry Tracing 集成
- 每次调用的 Token 计数（input/output）
- 调用链可视化（Agent → LLM → Tool → LLM → ...）
- 成本自动计算

**涉及文件:**
- `core/tracing.py` - Tracing 框架
- `core/callbacks.py` - 扩展 `MetricsCallback` 支持 Token 计数

---

### 9. 更多 Agent 模式
- **ReWOO Agent**：先规划所有工具调用，再批量执行
- **Reflexion Agent**：自反思 + 自我修正
- **Code Interpreter Agent**：动态执行代码

**涉及文件:**
- `agent/ReWOOAgent.py`
- `agent/ReflexionAgent.py`
- `agent/CodeInterpreterAgent.py`

---

### 10. 更多预置工具
- `Tool/builtin/code_interpreter.py` - 代码执行器
- `Tool/builtin/file_manager.py` - 文件管理
- `Tool/builtin/http_client.py` - HTTP 请求

---

### 11. V2 记忆性能优化
- 批量 embedding 管道优化
- 向量缓存层减少重复编码
- 异步并发存储写入

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

### 16. 高级 RAG 增强
- Hybrid Search（向量 + BM25 关键词）
- GraphRAG 集成
- Re-ranking 模块
- 语义分块策略优化

---

### 17. 分布式 Agent
支持在多台机器上运行 Agent 集群，任务自动分发与负载均衡。

---

### 18. 更多 LLM 提供商支持
- 讯飞星火
- 百川
- MiniMax

---

## ✅ 已完成

| 功能 | 完成日期 | 版本 |
|------|----------|------|
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
| ConversationSummaryMemory | 2026-01-19 | v1.0 |
| StructuredOutputAgent | 2026-01-19 | v1.0 |
| WebSearchTool | 2026-01-19 | v1.0 |
| CalculatorTool | 2026-01-19 | v1.0 |
| 单元测试 (52 tests) | 2026-01-19 | v1.0 |

---

## 贡献指南

如果你想贡献代码，请：
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交代码 (`git commit -m 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request
