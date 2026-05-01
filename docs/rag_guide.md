# RAG Guide

RAG 模块负责检索增强生成。它的目标不是替代 Tool 或 Memory，而是让 Agent 能把外部文档知识库按需检索出来，再以结构化方式喂给模型。

如果你在做：

- 企业知识问答
- 文档助手
- 代码库文档检索
- 长文档问答

RAG 通常比把全部资料直接塞进 prompt 更合理。

相关文档：

- [Context And Compaction Guide](./context_and_compaction_guide.md)
- [Tool System Guide](./tool_system_guide.md)
- [Memory System Guide](./memory_system_guide.md)

## 1. 核心对象

RAG 层的核心对象有：

- `RAGPipeline`
  - 统一编排“加载、分块、嵌入、存储、检索、生成”的总控。
- `DocumentLoader`
  - 把文件加载成 `Document`。
- `BaseChunker`
  - 文档分块器抽象。
- `BaseEmbedding`
  - 嵌入模型抽象。
- `BaseVectorStore`
  - 向量存储抽象。
- `BaseRetriever`
  - 检索器抽象。
- 可选增强：
  - `QueryTransformer`
  - rerank / hybrid / compression retriever

## 2. RAG 模块解决什么问题

如果你把大量外部文档直接放进 prompt，会遇到：

- token 太贵
- 上下文窗口不够
- 大部分内容其实这轮用不到
- cache 很难稳定

RAG 的基本思路是：

1. 先离线或预先把文档切块并建立检索索引
2. 每轮只检索与当前问题最相关的几个片段
3. 再把这些片段喂给模型

## 3. `RAGPipeline` 负责什么

`RAGPipeline` 是 RAG 的总控对象。它主要负责：

1. 接收：
   - `llm`
   - `embedding`
   - `vectorstore`
   - `loader`
   - `chunker`
   - `retriever`
   - `query_transformer`
2. 导入文档
3. 构建检索索引
4. 查询时检索相关片段
5. 用 prompt template 把上下文与问题组合起来
6. 调用 LLM 生成答案

也就是说，RAGPipeline 既能做 ingestion，也能做 query-time orchestration。

## 4. 主要组件逐项解释

### `DocumentLoader`

负责把各种文件读取成统一 `Document`。

当前实现支持很多格式，底层主要基于 MarkItDown 做统一转换。

适合：

- pdf
- doc/docx
- txt/md
- 表格
- 图片 OCR

### `BaseChunker`

负责把文档切成多个 `Document_Chunk`。

目标是：

- 每个块足够短，便于检索和注入 prompt
- 又不能短到丢失语义

### `BaseEmbedding`

负责把文本变成向量。

接口重点是：

- `embed_documents(texts)`
- `embed_query(text)`
- `dimension`

### `BaseVectorStore`

负责向量持久化和相似度检索。

主要能力：

- `add_documents`
- `similarity_search`
- `similarity_search_with_score`
- `delete`
- `clear`

### `BaseRetriever`

负责根据 query 找出相关 chunk。

它可以基于：

- 纯向量
- BM25
- hybrid
- rerank

## 5. 一次典型执行流程

### 导入阶段

1. `DocumentLoader` 读取文档
2. `Chunker` 切块
3. `Embedding` 生成向量
4. `VectorStore` 持久化

### 查询阶段

1. 用户提出问题
2. 可选 `query_transformer` 改写查询
3. `Retriever` 找相关 chunk
4. `RAGPipeline` 组装 `context + question`
5. 调用 LLM 生成答案

## 6. 一个最小可运行示例

```python
from easyagent import EasyLLM
from rag import RAGPipeline
from rag.embedding import OpenAIEmbedding
from rag.vectorstore import MemoryVectorStore

pipeline = RAGPipeline(
    llm=EasyLLM(),
    embedding=OpenAIEmbedding(),
    vectorstore=MemoryVectorStore(),
)

pipeline.ingest_from_path("./docs")
answer = pipeline.query("这个项目的 session 是怎么恢复的？")
```

## 7. `RAGPipeline` 的常用参数

### `llm`

最终回答问题的模型实例。

### `embedding`

文档和 query 的嵌入模型。

### `vectorstore`

存储 chunk 向量的后端。

### `loader`

文档加载器。默认会使用 `DocumentLoader`。

### `chunker`

分块器。决定每个 chunk 的粒度。

### `retriever`

检索器。若不传，会使用默认向量检索器。

### `query_transformer`

可选。适合做：

- query rewrite
- HyDE
- multi-query

### `prompt_template`

最终生成答案时使用的模板。通常需要包含：

- `{context}`
- `{question}`

### `k`

默认检索多少个 chunk。

## 8. 如何把 RAG 接到 Agent

RAG 一般有两种接法。

### 方式一：做成 context source

最推荐的方式。

适合：

- 你希望 RAG 像 memory 一样自动参与请求构建
- 你希望统一受 `ContextManager` 的预算控制

### 方式二：做成 Tool

适合：

- 你希望模型显式决定什么时候检索
- 你希望把检索动作暴露给用户和日志

通常不建议直接把 `RAGPipeline` 塞进 Agent，而是让它变成：

- `BaseContextSource`
- 或 Tool

## 9. RAG 和 Memory 的区别

两者都能提供额外上下文，但语义不同。

### Memory

- 偏会话/用户/长期积累
- 关注“应该记住什么”

### RAG

- 偏外部知识库
- 关注“这一轮应该检索什么”

一个成熟产品通常两者都会有。

## 10. RAG 和 cache 的关系

RAG 天然会让请求尾部更动态，因为检索结果每轮可能不同。

这意味着：

- RAG 更适合进入 dynamic tail 或 context source 层
- 不适合进入稳定 system core

所以正确目标不是“RAG 不影响 cache”，而是：

- 只让 RAG 影响尾部
- 不要污染稳定前缀

## 11. 推荐实践

### 先把 RAG 当作检索层，不要当作全能知识系统

它擅长检索相关片段，不擅长替你定义业务语义。

### 从小规模索引开始

先验证：

- 分块策略
- 检索质量
- prompt template

再考虑更复杂的 rerank / hybrid。

### 尽量把结果做成结构化上下文

而不是一股脑拼成无来源长文本。

### 对大文档库优先做离线 ingest

不要每轮现切现嵌入。

## 12. 常见坑

### 坑一：chunk 太大

检索到的片段太长，会增加无关噪音和 token 成本。

### 坑二：chunk 太小

语义被切碎，检索命中后也无法回答问题。

### 坑三：把 RAG 结果放进 system prompt

这会严重破坏稳定前缀和 cache。

### 坑四：检索到什么就全塞进去

应该控制 `k`，必要时做 rerank 或压缩。

### 坑五：把 RAG 和 memory 混成一个模块

两者职责不同，最好分别建模，再在 Context 层合流。
