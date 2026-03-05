# EasyAgent 待实现功能 (Roadmap)

本文档记录 EasyAgent 框架后续计划实现的功能。

---

## 🔴 高优先级 (P1)

### 1. V2 记忆系统与 Agent 集成
将 V2 多层记忆系统（EpisodicMemory、SemanticMemory、PerceptualMemory、WorkingMemory）正式集成到 Agent 框架中，使 Agent 可以直接使用多层记忆。

**涉及文件:**
- `core/agent.py` - BaseAgent 增加 V2 记忆系统支持
- `agent/ConversationalAgent.py` - 集成 V2 记忆
- 新增 `agent/MemoryAgent.py` - 专用记忆增强 Agent

---

### 2. MCP (Model Context Protocol) 支持
允许 Agent 连接外部 MCP 服务器，扩展工具生态。

```python
# 预期用法
from Tool.mcp import MCPClient

mcp_client = MCPClient("http://localhost:8080")
tools = mcp_client.get_tools()
registry.register_from_mcp(mcp_client)
```

**涉及文件:**
- `Tool/mcp.py` - MCP 客户端实现
- `Tool/ToolRegistry.py` - 支持从 MCP 注册工具

---

### 3. 异步 Agent 调用支持
支持异步调用 Agent 和 LLM，包括异步工具执行。

```python
async def main():
    response = await agent.ainvoke(query)
    async for chunk in agent.astream(query):
        print(chunk)
```

**涉及文件:**
- `core/llm.py` - 添加 `ainvoke`, `astream` 方法
- `agent/*.py` - 添加异步版本方法

---

### 4. 更多向量/图数据库支持
- Milvus VectorStore 实现
- Pinecone VectorStore 实现
- `memory/V2/Store/MilvusVectorStore.py`
- `memory/V2/Store/PineconeVectorStore.py`

---

## 🟡 中优先级 (P2)

### 5. AgentExecutor 统一执行器
标准化 Agent 执行流程，统一管理回调、错误处理、重试。

```python
executor = AgentExecutor(
    agent=agent,
    callbacks=[logging_callback],
    max_retries=3,
    timeout=60
)
result = executor.run(query)
```

**涉及文件:**
- `core/executor.py`

---

### 6. MultiAgentOrchestrator 多 Agent 协作
支持多个 Agent 协作完成复杂任务。

```python
orchestrator = MultiAgentOrchestrator([
    ("researcher", researcher_agent),
    ("writer", writer_agent),
    ("reviewer", reviewer_agent),
])
result = orchestrator.run("写一篇关于 AI 的报告")
```

**涉及文件:**
- `agent/orchestrator.py`
- `agent/communication.py`

---

### 7. 更多预置工具
- `Tool/builtin/code_interpreter.py` - 代码执行器
- `Tool/builtin/file_manager.py` - 文件管理
- `Tool/builtin/http_client.py` - HTTP 请求

---

### 8. Agent 持久化
支持保存和加载 Agent 状态（包括记忆、配置）。

```python
agent.save("./agent_state.json")
agent = BasicAgent.load("./agent_state.json")
```

---

### 9. V2 记忆性能优化
- 批量 embedding 管道优化
- 向量缓存层减少重复编码
- 异步并发存储写入

---

## 🟢 低优先级 (P3)

### 10. Web UI 管理界面
基于 Gradio/Streamlit 的可视化管理界面。

---

### 11. 分布式 Agent
支持在多台机器上运行 Agent 集群。

---

### 12. 更多 LLM 提供商支持
- 讯飞星火
- 百川
- MiniMax

---

### 13. 视频模态支持
为 PerceptualMemory 添加视频编码和检索能力。

---

## ✅ 已完成

| 功能 | 完成日期 | 版本 |
|------|----------|------|
| **V2 感知记忆 (PerceptualMemory)** | 2026-03 | v2.0-dev |
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
| **Provider 适配器模式** | 2026-01-19 | v1.1 |
| ConversationSummaryMemory | 2026-01-19 | v1.0 |
| StructuredOutputAgent | 2026-01-19 | v1.0 |
| WebSearchTool | 2026-01-19 | v1.0 |
| CalculatorTool | 2026-01-19 | v1.0 |
| Callbacks 回调系统 | 2026-01-19 | v1.0 |
| 单元测试 (52 tests) | 2026-01-19 | v1.0 |

---

## 贡献指南

如果你想贡献代码，请：
1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交代码 (`git commit -m 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request
