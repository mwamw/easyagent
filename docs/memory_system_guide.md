# 可替换记忆模块

## 框架契约

BaseMemory 表示一个完整记忆机制，不要求统一 CRUD、分类或存储。
Memory 可以提供 get_prompt_blocks、get_tools、get_hooks；这些能力均可为空。
bind/unbind 和 services 由基类提供；同一有状态实例只能绑定一个 Agent。

```python
from easyagent.memory import BaseMemory
from easyagent.prompting import PromptBlock

class ApplicationMemory(BaseMemory):
    def __init__(self, tools):
        self.tools = list(tools)

    def get_prompt_blocks(self, context):
        return [PromptBlock("memory.application", "Use the supplied memory tools when prior knowledge is needed.")]

    def get_tools(self):
        return self.tools

agent.with_memory(ApplicationMemory(memory_tools))
```

机制说明使用 PromptBlock，不需要 MemoryDescriptor。
模型搜索或保存通过该实现自己的 Tool，结果正常走 ToolResult，不重复注入。
模块规则与产品提示词组合；关闭框架默认规则不会自动关闭模块说明。
完全接管提示词可设置 composer.include_module_blocks=False。

## 动态注入是可选能力

```python
from easyagent.hooks import BaseHook
from easyagent.metamessages import MetaMessage, MetaMessageLifecycle

class RecallHook(BaseHook):
    def __init__(self, memory):
        self.memory = memory

    async def abefore_context(self, payload):
        text = await self.memory.client.recall(payload["query"])
        if text:
            self.memory.services.metamessages.emit(MetaMessage(
                name="recall", content=text,
                metadata={"source": "memory.application"},
                lifecycle=MetaMessageLifecycle.REQUEST,
            ))
```

Memory 自己检索、指定内容和来源；MetaMessage 不理解来源业务。
异步专用 Hook 必须使用 ainvoke/astream。同步 Hook 用于轻量计算，
阻塞 I/O 应提供异步实现或显式安全线程桥接。

## 内置 LayeredMemory

```python
from easyagent.memory import LayeredMemory
agent.with_memory(LayeredMemory(), owned=True)
```

它复用现有 V2 多层算法，默认只有 working memory，默认不自动召回。
模型通过六个内置记忆工具读写。分类、MemoryItem、重要性和遗忘策略
属于这一方案，不是 BaseMemory 的强制契约。
内部 MemoryManage 和 V2 文件名保留，但不再作为 Agent 的安装接口。

会话保存可包含 working memory；不会复制外部向量库、图数据库。
记忆工具使用当前 Agent 的 session_id，不再自行生成另一套会话 ID。
默认 owned=False：框架解除注册，但不关闭外部共享资源。
替换时显式使用 replace=True；原有用户工具发生名称冲突时会报错。
拥有异步资源的模块可实现 aclose，并使用 await agent.aclose() 释放；
内置实现只关闭自己创建的后端，不关闭调用方注入的共享后端。

## 公共基础设施

easyagent.storage 提供 DocumentStore、VectorStore、GraphStore 和 EmbeddingModel。
Memory 的领域 repository 负责 MemoryItem 转换，RAG 的 ChunkIndex 负责文档转换。
公共后端不导入 MemoryItem、MemoryType 或 Document_Chunk。
旧 SQLite 记忆数据使用 python -m easyagent.migrate memory-sqlite 显式复制迁移。
