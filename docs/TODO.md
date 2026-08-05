# EasyAgent 待实现功能 (Roadmap)

本文档记录 EasyAgent 框架后续计划实现的功能。

---

## 🔴 高优先级 (P1)

### 1. Claude Thinking 模型支持
支持 Claude Thinking 模型的多轮工具调用（需要在对话历史中保留 thinking blocks）。

```python
# 需要处理 thinking blocks 的保留
agent = BasicAgent(llm=EasyLLM(model="claude-sonnet-4-5-thinking"))
```

**涉及文件:**
- `core/providers/anthropic_provider.py` - 处理 thinking blocks
- `agent/BasicAgent.py` - 保留 thinking 历史

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
```

**涉及文件:**
- `agent/BasicAgent.py` - 增加 `stream_invoke_with_tool` 方法
- `core/llm.py` - 增加 `stream_invoke_with_tools` 方法

---

### 3. 异步调用支持 (Async Support)
支持异步调用 Agent 和 LLM。

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

## 🟡 中优先级 (P2)

### 4. AgentExecutor 统一执行器
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

### 5. MultiAgentOrchestrator 多 Agent 协作
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

### 6. 更多预置工具
- `Tool/builtin/code_interpreter.py` - 代码执行器
- `Tool/builtin/file_manager.py` - 文件管理
- `Tool/builtin/http_client.py` - HTTP 请求

---

### 7. Agent 持久化
支持保存和加载 Agent 状态（包括记忆、配置）。

```python
agent.save("./agent_state.json")
agent = BasicAgent.load("./agent_state.json")
```

---

## 🟢 低优先级 (P3)

### 8. Web UI 管理界面
基于 Gradio/Streamlit 的可视化管理界面。

---

### 9. 分布式 Agent
支持在多台机器上运行 Agent 集群。

---

### 10. 更多 LLM 提供商支持
- Anthropic Claude
- 讯飞星火
- 百川
- MiniMax

---

### 11. 向量数据库扩展
- Milvus 支持
- Pinecone 支持
- Weaviate 支持

---

## ✅ 已完成

| 功能 | 完成日期 | 版本 |
|------|----------|------|
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
