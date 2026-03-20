# EasyAgent

一个功能完整、可扩展的 AI Agent 开发框架，支持多种 Agent 类型、多层记忆系统、RAG、工具调用等。

## 📦 特性

- **多种 Agent 类型**：BasicAgent、ReactAgent、PlanningAgent、ConversationalAgent、RAGAgent、StructuredOutputAgent
- **V2 多层记忆系统**：情景记忆、语义记忆、感知记忆、工作记忆（仿人类认知架构）
- **多模态感知**：支持文本、图像 (CLIP)、音频 (CLAP) 的编码与跨模态检索
- **知识图谱**：LLM 实体关系提取 + Neo4j 图数据库存储
- **RAG 支持**：文档加载、文本分割、向量检索
- **输出解析**：JSON、Pydantic 模型解析
- **提示词模板**：可复用的提示词管理
- **多模型支持**：OpenAI、Google Gemini、Anthropic Claude、DeepSeek、通义千问等 (Provider 适配器模式)
- **异步支持**：批量添加记忆与异步搜索

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/EasyAgent.git
cd EasyAgent

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 基础使用

```python
from agent import BasicAgent
from core.llm import EasyLLM

# 创建 LLM 和 Agent
llm = EasyLLM(model="gemini-2.5-flash", provide="google")
agent = BasicAgent(name="assistant", llm=llm)

# 对话
response = agent.invoke("你好，请介绍一下你自己")
print(response)
```

### 带工具调用

```python
from agent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from pydantic import BaseModel, Field

# 创建工具注册表
registry = ToolRegistry()

# 定义工具
class SearchParams(BaseModel):
    query: str = Field(description="搜索关键词")

@registry.tool("search", "搜索引擎", SearchParams)
def search(query: str) -> str:
    return f"搜索结果: {query}"

# 创建带工具的 Agent
agent = BasicAgent(
    name="tool_agent",
    llm=llm,
    enable_tool=True,
    tool_registry=registry
)

response = agent.invoke("帮我搜索最新的AI新闻")
```

### MCP 工具集成

```python
from agent import BasicAgent
from core.llm import EasyLLM
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_mcp_tools

llm = EasyLLM(model="gemini-2.5-flash", provide="google")
registry = ToolRegistry()

# 示例 1: 通过本地 Python MCP Server 脚本接入（stdio）
mcp_manager = register_mcp_tools(
    registry=registry,
    server_source=["python", "./examples/mcp_server.py"],
    tool_prefix="mcp_",
)

agent = BasicAgent(
    name="mcp_agent",
    llm=llm,
    enable_tool=True,
    tool_registry=registry,
)

print(agent.invoke("请调用 mcp_calc 计算 12*7"))

# 结束时关闭连接
mcp_manager.close()
```

说明：

- `register_mcp_tools(...)` 会自动从 MCP Server 拉取 `list_tools`，并注册到 `ToolRegistry`
- 支持 `stdio/http/sse/FastMCP(memory)` 传输
- 需要安装可选依赖：`pip install fastmcp>=2.0.0`

你也可以使用更简洁的语法糖写法：

```python
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import mcptool

registry = ToolRegistry()

mcp_tool = mcptool(
    server_source=[
        "npx",
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/wxd/LLM/EasyAgent",
    ],
    tool_prefix="fs_",
)

# 等价于 mcp_tool.register_to_registry(registry)
registry.registry(mcp_tool)
```

注意：上面的变量应是 `mcp_tool`（你示例里的 `mcptool` 如果是函数名，需先赋值给变量再注册）。

### 对话记忆

```python
from agent import ConversationalAgent
from memory import MemoryManage, MemoryConfig

memory_manage = MemoryManage(
    config=MemoryConfig(),
    user_id="user1",
    enable_working=True,
    enable_episodic=False,
    enable_semantic=False,
    enable_perceptual=False,
)

agent = ConversationalAgent(
    name="chatbot",
    llm=llm,
    memory_manage=memory_manage,
)

agent.invoke("我叫张三，今年25岁")
agent.invoke("你还记得我的名字和年龄吗？")  # 能记住上下文
```

### RAG 知识问答

```python
from agent import RAGAgent
from rag import TextLoader, RecursiveCharacterTextSplitter
from rag import ChromaVectorStore, VectorStoreRetriever

# 加载和分割文档
docs = TextLoader("knowledge.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(docs)

# 创建向量存储
store = ChromaVectorStore(collection_name="knowledge_base")
store.add_documents(chunks)

# 创建检索器和 RAG Agent
retriever = VectorStoreRetriever(vectorstore=store, k=5)
agent = RAGAgent(name="rag_agent", llm=llm, retriever=retriever)

answer = agent.invoke("什么是机器学习？")
```

### V2 情景记忆使用

```python
from memory.V2.EpisodicMemory import EpisodicMemory
from memory.V2.BaseMemory import MemoryConfig, MemoryItem
from memory.V2.Store.SQLiteDocumentStore import SQLiteDocumentStore
from memory.V2.Store.QdrantVectorStore import QdrantVectorStore
from memory.V2.Embedding.HuggingfaceEmbeddingModel import HuggingfaceEmbeddingModel
from datetime import datetime

# 初始化存储和嵌入模型
doc_store = SQLiteDocumentStore(db_path="memory.db")
vec_store = QdrantVectorStore(way="memory", collection_name="episodic")
embedding = HuggingfaceEmbeddingModel()

# 创建情景记忆
episodic = EpisodicMemory(
    config=MemoryConfig(),
    document_store=doc_store,
    vector_store=vec_store,
    embedding_model=embedding
)

# 添加记忆
item = MemoryItem(
    id="ep_001", content="用户今天讨论了GraphRAG的技术细节",
    type="episodic", user_id="user1",
    timestamp=datetime.now(), importance=0.8, metadata={}
)
episodic.add_memory(item)

# 搜索记忆
results = episodic.search_memory("GraphRAG", limit=5)
```

### V2 语义记忆 + 知识图谱

```python
from memory.V2.SemanticMemory import SemanticMemory
from memory.V2.Store.Neo4jGraphStore import Neo4jGraphStore
from memory.V2.Extractor.Extractor import Extractor

# 初始化图数据库和提取器
graph_store = Neo4jGraphStore(uri="bolt://localhost:7687", username="neo4j", password="xxx")
extractor = Extractor(llm=llm)

# 创建语义记忆
semantic = SemanticMemory(
    memory_config=MemoryConfig(),
    vector_store=vec_store,
    graph_store=graph_store,
    extractor=extractor,
    embedding_model=embedding
)

# 添加记忆 → 自动提取实体和关系 → 存入图数据库
semantic.add_memory(item)

# 语义搜索 (向量 + 图谱混合排序)
results = semantic.search_memory("GraphRAG相关技术")
```

## 📁 项目结构

```
EasyAgent/
├── agent/                         # Agent 实现
│   ├── BasicAgent.py              # 基础工具调用 Agent
│   ├── ReactAgent.py              # 显式思考链 Agent (ReAct)
│   ├── PlanningAgent.py           # 规划执行 Agent
│   ├── ConversationalAgent.py     # 对话记忆 Agent
│   ├── RAGAgent.py                # 检索增强 Agent
│   └── StructuredOutputAgent.py   # 结构化输出 Agent
├── memory/                        # 记忆系统
│   ├── base.py                    # V1 记忆基类
│   ├── buffer.py                  # V1 对话缓冲记忆
│   ├── vector.py                  # V1 向量语义记忆
│   ├── summary.py                 # V1 对话摘要记忆
│   └── V2/                        # V2 多层记忆系统 ⭐
│       ├── BaseMemory.py          # V2 记忆基类 (MemoryItem, MemoryConfig)
│       ├── EpisodicMemory.py      # 情景记忆 (事件/会话)
│       ├── SemanticMemory.py      # 语义记忆 (知识图谱 + 向量)
│       ├── PerceptualMemory.py    # 感知记忆 (多模态)
│       ├── WorkingMemory.py       # 工作记忆 (内存短期)
│       ├── Store/                 # 存储后端
│       │   ├── DocumentStore.py       # 文档存储接口
│       │   ├── SQLiteDocumentStore.py # SQLite 实现
│       │   ├── VectorStore.py         # 向量存储接口
│       │   ├── QdrantVectorStore.py   # Qdrant 实现
│       │   ├── GraphStore.py          # 图存储接口 (Entity, Relation)
│       │   └── Neo4jGraphStore.py     # Neo4j 实现
│       ├── Embedding/             # 嵌入模型
│       │   ├── BaseEmbeddingModel.py      # 嵌入模型接口
│       │   └── HuggingfaceEmbeddingModel.py # SentenceTransformer 实现
│       └── Extractor/             # 实体关系提取
│           └── Extractor.py       # LLM 提取 + 验证
├── core/                          # 核心模块
│   ├── llm.py                     # EasyLLM 统一接口
│   ├── agent.py                   # BaseAgent 基类
│   ├── Message.py                 # 消息类型
│   ├── Config.py                  # 配置管理
│   ├── Exception.py               # 异常定义
│   ├── callbacks.py               # 回调系统
│   └── providers/                 # LLM Provider 适配器
│       ├── base.py                    # Provider 基类
│       ├── openai_provider.py         # OpenAI 兼容
│       ├── google_provider.py         # Google Gemini
│       └── anthropic_provider.py      # Anthropic Claude
├── Tool/                          # 工具模块
│   ├── BaseTool.py                # 工具基类
│   ├── ToolRegistry.py            # 工具注册表
│   ├── AsyncToolExecutor.py       # 异步工具执行器
│   ├── memory_tools.py            # 记忆工具
│   └── builtin/                   # 预置工具
│       ├── calculator.py          # 安全计算器
│       └── search.py              # 搜索工具
├── output/                        # 输出解析器
│   ├── base.py                    # BaseOutputParser
│   ├── json_parser.py             # JSON 解析器
│   └── pydantic_parser.py         # Pydantic 解析器
├── prompt/                        # 提示词模板
│   ├── template.py                # PromptTemplate / ChatPromptTemplate
│   └── defaults.py                # 预置提示词
├── rag/                           # RAG 模块
│   ├── document.py                # 文档定义
│   ├── loader.py                  # 文档加载器 (文本/PDF/目录)
│   ├── splitter.py                # 文本分割器 (递归/Token)
│   ├── vectorstore.py             # ChromaDB 向量存储
│   └── retriever.py               # 检索器 (向量/多查询)
└── test/                          # 测试
    ├── test_episodememory.py       # 情景记忆测试
    ├── test_semanticmemory.py     # 语义记忆测试
    ├── test_PerceptualMemory.py   # 感知记忆测试
    ├── test_working_memory.py     # 工作记忆测试
    ├── test_Neo4jStore.py         # Neo4j 存储测试
    ├── test_Qdramt.py             # Qdrant 存储测试
    ├── test_sqlite.py             # SQLite 存储测试
    ├── test_Extractor.py          # 提取器测试
    └── ...                        # 其他测试
```

## 🔧 配置

在 `.env` 文件中配置：

```env
# LLM 配置
LLM_MODEL_ID=gemini-2.5-flash
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# 可选：其他服务
SERPAPI_API_KEY=your_serpapi_key

# V2 记忆系统 (可选)
IMAGE_ENCODER=openai/clip-vit-base-patch32
AUDIO_ENCODER=laion/clap-htsat-unfused
```

## 📚 Agent 类型说明

| Agent | 用途 | 特点 |
|-------|------|------|
| BasicAgent | 通用对话和工具调用 | 支持 Function Calling |
| ReactAgent | 复杂推理任务 | 显式思考链 (Thought → Action → Observation) |
| PlanningAgent | 多步骤任务 | 任务分解和逐步执行 |
| ConversationalAgent | 多轮对话 | 集成记忆系统 |
| RAGAgent | 知识问答 | 检索增强生成 |
| StructuredOutputAgent | 信息提取 | Pydantic Schema 强制输出 |

## 🧠 V2 记忆系统架构

仿人类认知的四层记忆架构：

| 记忆类型 | 说明 | 持久化 | 存储后端 |
|----------|------|--------|----------|
| **WorkingMemory** | 短期工作记忆 | ❌ 仅内存 | 内存堆 |
| **EpisodicMemory** | 情景记忆 (事件/会话) | ✅ | SQLite + Qdrant |
| **SemanticMemory** | 语义记忆 (知识/概念) | ✅ | Qdrant + Neo4j |
| **PerceptualMemory** | 感知记忆 (多模态) | ✅ | SQLite + Qdrant (多 collection) |

### 核心能力

- **批量/异步操作**：`add_memories_batch` / `add_memory_async` / `search_memory_async`
- **遗忘机制**：按时间 / 重要性 / 容量策略自动遗忘
- **持久化恢复**：`load_from_store()` 从存储层重建缓存
- **数据同步**：`sync_stores()` 检测并修复缓存与存储的不一致
- **模式发现**：`find_patterns()` 基于词频/TF-IDF/语义聚类发现行为模式
- **知识图谱**：自动提取实体关系 → 图谱增强语义搜索
- **多模态**：CLIP (图像) / CLAP (音频) 编码与跨模态检索

## 🛠️ 依赖

### 基础依赖
- Python >= 3.10
- openai >= 1.0.0
- pydantic >= 2.0.0

### RAG 模块
- chromadb >= 0.4.0
- pypdf >= 3.0.0

### V2 记忆系统
- sentence-transformers (嵌入模型)
- qdrant-client (向量存储)
- neo4j (图数据库，语义记忆需要)
- transformers + torch (多模态编码，感知记忆需要)
- jieba / scikit-learn (模式发现)

### 搜索工具 (可选)
- duckduckgo-search 或 google-search-results

## 📄 许可证

MIT License
