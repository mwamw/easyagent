# EasyAgent 更新日志 (Changelog)

本文档记录 EasyAgent 框架各版本的更新内容。

---

## [v2.0.0-dev] - 2026-02 ~ 2026-03 (开发中)

### 🚀 新特性

#### V2 多层记忆系统
全新仿人类认知的多层记忆架构，位于 `memory/V2/` 目录。

**记忆类型：**
- `memory/V2/BaseMemory.py` - V2 记忆基类，定义 `MemoryItem`、`MemoryConfig`、`MemoryType`、`ForgetType`
- `memory/V2/EpisodicMemory.py` - 情景记忆：事件/会话级记忆，支持时间线、模式发现
- `memory/V2/SemanticMemory.py` - 语义记忆：知识图谱 + 向量混合检索
- `memory/V2/PerceptualMemory.py` - 感知记忆：多模态（文本/图像/音频）编码与跨模态检索
- `memory/V2/WorkingMemory.py` - 工作记忆：内存短期记忆，优先级堆管理

**存储后端：**
- `memory/V2/Store/DocumentStore.py` - 文档存储抽象接口
- `memory/V2/Store/SQLiteDocumentStore.py` - SQLite 文档存储实现
- `memory/V2/Store/VectorStore.py` - 向量存储抽象接口
- `memory/V2/Store/QdrantVectorStore.py` - Qdrant 向量存储实现（支持内存/本地/云端模式）
- `memory/V2/Store/GraphStore.py` - 图存储抽象接口（含 Entity, Relation 数据类）
- `memory/V2/Store/Neo4jGraphStore.py` - Neo4j 图存储实现

**嵌入层：**
- `memory/V2/Embedding/BaseEmbeddingModel.py` - 嵌入模型抽象接口
- `memory/V2/Embedding/HuggingfaceEmbeddingModel.py` - SentenceTransformer 实现

**提取器：**
- `memory/V2/Extractor/Extractor.py` - LLM 实体关系提取器（两阶段：提取 + 验证）

#### 核心能力
- **批量/异步**：所有 V2 记忆类型支持 `add_memories_batch`、`add_memory_async`、`search_memory_async`
- **遗忘机制**：按时间 / 重要性 / 容量的多策略遗忘
- **持久化**：`load_from_store()` / `sync_stores()` 支持从存储层恢复缓存和数据同步
- **模式发现**：EpisodicMemory 的 `find_patterns()` 基于 jieba 分词、TF-IDF、语义聚类
- **多模态编码**：CLIP (图像)、CLAP (音频) 编码与跨模态文本检索
- **知识图谱搜索**：SemanticMemory 的混合排序（向量相似度 + 图谱上下文相关性加权）

#### 工具增强
- `Tool/AsyncToolExecutor.py` - 异步工具执行器
- `Tool/memory_tools.py` - 记忆相关工具

#### 测试覆盖
- `test/test_episodememory.py` - 情景记忆完整测试
- `test/test_semanticmemory.py` - 语义记忆完整测试
- `test/test_PerceptualMemory.py` - 感知记忆完整测试
- `test/test_working_memory.py` - 工作记忆完整测试
- `test/test_Neo4jStore.py` - Neo4j 存储测试
- `test/test_Qdramt.py` - Qdrant 存储测试
- `test/test_sqlite.py` - SQLite 存储测试
- `test/test_Extractor.py` - 提取器测试
- `test/test_async_tool_executor.py` - 异步执行器测试

---

## [v1.1.0] - 2026-01-19

### 🚀 新特性

#### Provider 适配器模式
- 新增 `core/providers/` 目录，实现 LLM Provider 适配器模式
- 支持自动检测模型类型并使用对应的 Provider
- 统一工具结果消息格式处理

**新增文件：**
- `core/providers/base.py` - Provider 抽象基类
- `core/providers/openai_provider.py` - OpenAI API 兼容 Provider
- `core/providers/google_provider.py` - Google Gemini Provider
- `core/providers/anthropic_provider.py` - Anthropic Claude Provider
- `core/providers/__init__.py` - 工厂函数和自动检测

**改进：**
- `EasyLLM` 重构为使用 Provider 模式
- 新增 `llm.format_tool_result()` 方法，自动格式化工具结果
- `BasicAgent` 简化，移除手动消息类选择逻辑

### 📝 文档
- 新增 `docs/CHANGELOG.md` 更新日志
- 更新 `docs/TODO.md` 待实现功能列表

---

## [v1.0.0] - 2026-01-19

### 🎉 首次发布

#### 核心模块
- `core/llm.py` - EasyLLM 统一 LLM 接口
- `core/agent.py` - BaseAgent 基类
- `core/Message.py` - 消息类型定义
- `core/Config.py` - 配置管理
- `core/Exception.py` - 异常定义
- `core/callbacks.py` - 回调系统

#### Agent 实现
- `agent/BasicAgent.py` - 基础智能体，支持工具调用
- `agent/ReactAgent.py` - ReAct 模式智能体
- `agent/RAGAgent.py` - 检索增强生成智能体
- `agent/ConversationalAgent.py` - 对话智能体
- `agent/PlanningAgent.py` - 规划智能体
- `agent/StructuredOutputAgent.py` - 结构化输出智能体

#### 工具系统
- `Tool/BaseTool.py` - 工具基类
- `Tool/ToolRegistry.py` - 工具注册表
- `Tool/builtin/calculator.py` - 安全计算器工具
- `Tool/builtin/search.py` - 网络搜索工具 (SerpAPI/DuckDuckGo)

#### V1 记忆系统
- `memory/base.py` - 记忆基类
- `memory/buffer.py` - 对话缓冲记忆
- `memory/vector.py` - 向量记忆
- `memory/summary.py` - 对话摘要记忆

#### 输出解析
- `output/base.py` - 解析器基类
- `output/json_parser.py` - JSON 解析器
- `output/pydantic_parser.py` - Pydantic 模型解析器

#### RAG 模块
- `rag/document.py` - 文档定义
- `rag/loader.py` - 文档加载器
- `rag/splitter.py` - 文本分割器
- `rag/vectorstore.py` - 向量存储
- `rag/retriever.py` - 检索器

#### 回调系统
- `BaseCallback` - 回调基类
- `LoggingCallback` - 日志回调
- `StreamingCallback` - 流式输出回调
- `MetricsCallback` - 指标收集回调
- `CallbackManager` - 回调管理器

#### 测试
- 52 个单元测试覆盖核心功能

---

## 版本规范

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范：

- **主版本号 (MAJOR)**：不兼容的 API 修改
- **次版本号 (MINOR)**：向后兼容的功能新增
- **修订号 (PATCH)**：向后兼容的问题修正

---

## 图例

- 🚀 新特性
- 🐛 Bug 修复
- 📝 文档更新
- ⚡ 性能优化
- 🔧 配置变更
- ⚠️ 破坏性变更
