# EasyAgent 框架 Walkthrough

## 完成概述

EasyAgent 框架已从 v1.0 基础版本发展到 v2.0-dev，新增了完整的仿人类认知多层记忆系统。

---

## 项目架构

```
EasyAgent/
├── agent/                         # 6 种 Agent 类型
│   ├── BasicAgent.py              # 工具调用 Agent
│   ├── ReactAgent.py              # ReAct 思考链
│   ├── PlanningAgent.py           # 规划执行
│   ├── ConversationalAgent.py     # 对话记忆
│   ├── RAGAgent.py                # 检索增强
│   └── StructuredOutputAgent.py   # 结构化输出
├── memory/V2/                     # V2 多层记忆系统 ⭐
│   ├── BaseMemory.py              # 基类 (MemoryItem, MemoryConfig)
│   ├── EpisodicMemory.py          # 情景记忆 (~980 行)
│   ├── SemanticMemory.py          # 语义记忆 (~920 行)
│   ├── PerceptualMemory.py        # 感知记忆 (~830 行)
│   ├── WorkingMemory.py           # 工作记忆 (~315 行)
│   ├── Store/                     # 三种存储后端
│   │   ├── SQLiteDocumentStore.py
│   │   ├── QdrantVectorStore.py
│   │   └── Neo4jGraphStore.py
│   ├── Embedding/                 # 嵌入模型
│   │   └── HuggingfaceEmbeddingModel.py
│   └── Extractor/                 # 实体关系提取
│       └── Extractor.py
├── core/                          # 核心 + Provider 适配器
├── Tool/                          # 工具系统 + 异步执行器
├── output/                        # 输出解析器
├── prompt/                        # 提示词模板
├── rag/                           # RAG 模块
└── test/                          # 22+ 测试文件
```

---

## V2 记忆系统详解

### 四层记忆架构

| 层 | 类 | 持久化 | 存储 | 核心能力 |
|----|-----|--------|------|----------|
| 工作 | `WorkingMemory` | ❌ | 内存堆 | 优先级排序、自动过期清理 |
| 情景 | `EpisodicMemory` | ✅ | SQLite + Qdrant | 时间线、模式发现、会话管理 |
| 语义 | `SemanticMemory` | ✅ | Qdrant + Neo4j | 知识图谱、实体提取、混合排序 |
| 感知 | `PerceptualMemory` | ✅ | SQLite + Qdrant(多) | CLIP/CLAP 多模态编码、跨模态检索 |

### 所有记忆类型共有能力
- `add_memory` / `add_memories_batch` - 添加记忆
- `remove_memory` / `update_memory` - 修改/删除
- `search_memory` - 混合检索 (向量 + 时间衰减 + 重要性加权)
- `forget(strategy)` - 按时间/重要性/容量遗忘
- `load_from_store()` - 从持久化层恢复缓存
- `sync_stores()` - 多层存储数据一致性同步
- 异步版本：`add_memory_async` / `search_memory_async`

### SemanticMemory 特色
- 添加记忆时自动通过 `Extractor` 提取实体和关系
- 搜索时结合向量相似度和图谱上下文进行混合排序
- 支持实体搜索、关联实体查询

### PerceptualMemory 特色
- 每个模态独立 VectorStore (text/image/audio)
- 文本查询自动转换为 CLIP/CLAP 嵌入进行跨模态检索
- 自动检测查询模态 (文件后缀名判断)

---

## Agent 类型汇总

| Agent | 用途 | 特点 |
|-------|------|------|
| BasicAgent | 通用对话和工具调用 | Function Calling |
| ReactAgent | 复杂推理任务 | Thought → Action → Observation |
| PlanningAgent | 多步骤任务 | 任务分解和逐步执行 |
| ConversationalAgent | 多轮对话 | 集成 V2 MemoryManage（可配置 Working/Episodic/Semantic） |
| RAGAgent | 知识问答 | 检索增强生成 |
| StructuredOutputAgent | 信息提取 | Pydantic Schema 强制输出 |

---

## 依赖安装

```bash
# 基础
pip install openai pydantic python-dotenv

# RAG
pip install chromadb pypdf

# V2 记忆系统
pip install sentence-transformers qdrant-client
pip install neo4j                     # 语义记忆
pip install transformers torch        # 多模态感知记忆
pip install jieba scikit-learn        # 模式发现

# 搜索工具 (可选)
pip install duckduckgo-search
```

---

## 测试覆盖

```
test/
├── test_episodememory.py      # 情景记忆完整测试
├── test_semanticmemory.py     # 语义记忆完整测试
├── test_PerceptualMemory.py   # 感知记忆完整测试
├── test_working_memory.py     # 工作记忆完整测试
├── test_Neo4jStore.py         # Neo4j 存储测试
├── test_Qdramt.py             # Qdrant 存储测试
├── test_sqlite.py             # SQLite 存储测试
├── test_Extractor.py          # 提取器测试
├── test_async_tool_executor.py # 异步执行器
├── test_basicagent.py         # Agent 测试
├── test_memory.py             # V1 记忆测试
├── test_output.py             # 输出解析器
├── test_callbacks.py          # 回调系统
└── test_builtin_tools.py      # 预置工具
```
