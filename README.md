# EasyAgent

一个功能完整、可扩展的 AI Agent 开发框架。

## 📦 特性

- **多种 Agent 类型**：BasicAgent、ReactAgent、PlanningAgent、ConversationalAgent、RAGAgent
- **记忆系统**：对话缓冲记忆、向量语义记忆
- **RAG 支持**：文档加载、文本分割、向量检索
- **输出解析**：JSON、Pydantic 模型解析
- **提示词模板**：可复用的提示词管理
- **多模型支持**：OpenAI、Google Gemini、DeepSeek、通义千问等

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/EasyAgent.git
cd EasyAgent

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp test/.env.example test/.env
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

### 对话记忆

```python
from agent import ConversationalAgent
from memory import ConversationBufferMemory

memory = ConversationBufferMemory(max_messages=20)
agent = ConversationalAgent(name="chatbot", llm=llm, memory=memory)

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

## 📁 项目结构

```
EasyAgent/
├── agent/                 # Agent 实现
│   ├── BasicAgent.py      # 基础工具调用 Agent
│   ├── ReactAgent.py      # 显式思考链 Agent
│   ├── PlanningAgent.py   # 规划执行 Agent
│   ├── ConversationalAgent.py  # 对话记忆 Agent
│   └── RAGAgent.py        # 检索增强 Agent
├── memory/                # 记忆系统
│   ├── base.py            # 抽象基类
│   ├── buffer.py          # 对话缓冲记忆
│   └── vector.py          # 向量语义记忆
├── output/                # 输出解析器
│   ├── base.py            # 抽象基类
│   ├── json_parser.py     # JSON 解析器
│   └── pydantic_parser.py # Pydantic 解析器
├── prompt/                # 提示词模板
│   ├── template.py        # 模板类
│   └── defaults.py        # 预置模板
├── rag/                   # RAG 模块
│   ├── document.py        # 文档类
│   ├── loader.py          # 文档加载器
│   ├── splitter.py        # 文本分割器
│   ├── vectorstore.py     # 向量存储
│   └── retriever.py       # 检索器
├── core/                  # 核心模块
│   ├── llm.py             # LLM 封装
│   ├── Message.py         # 消息类型
│   ├── Config.py          # 配置管理
│   └── Exception.py       # 异常类
└── Tool/                  # 工具模块
    ├── BaseTool.py        # 工具基类
    └── ToolRegistry.py    # 工具注册表
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
```

## 📚 Agent 类型说明

| Agent | 用途 | 特点 |
|-------|------|------|
| BasicAgent | 通用对话和工具调用 | 支持 Function Calling |
| ReactAgent | 复杂推理任务 | 显式思考链 |
| PlanningAgent | 多步骤任务 | 任务分解和逐步执行 |
| ConversationalAgent | 多轮对话 | 集成记忆系统 |
| RAGAgent | 知识问答 | 检索增强生成 |

## 🛠️ 依赖

- Python >= 3.10
- openai >= 1.0.0
- pydantic >= 2.0.0
- chromadb >= 0.4.0 (RAG 功能)
- pypdf >= 3.0.0 (PDF 加载)

## 📄 许可证

MIT License
