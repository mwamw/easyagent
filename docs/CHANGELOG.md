# EasyAgent 更新日志 (Changelog)

本文档记录 EasyAgent 框架各版本的更新内容。

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

#### 记忆系统
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
- `test/test_memory.py`
- `test/test_output.py`
- `test/test_callbacks.py`
- `test/test_builtin_tools.py`

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
