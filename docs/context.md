# Context 模块使用说明

本文档说明 EasyAgent 中 `context` 模块的定位、默认行为，以及如何做自定义扩展。

核心目标只有一件事：

- 在 token 预算内，为当前请求构建一份适合喂给 LLM 的 `messages`

当前实现里，Agent 可以在 history 超预算时直接保存压缩后的 history。  
正式会话历史和调试 trace 是两层不同的状态：

- `agent.history`: 当前继续对话要使用的正式会话消息，必要时可能已经被压缩
- `agent.trace_history`: 完整执行轨迹

---

## 1. 模块结构

`context` 模块主要由四层组成：

1. `ContextSource`
- 负责从不同来源取上下文
- 例如：
  - `HistoryContextSource`
  - `RAGContextSource`
  - `MemoryContextSource`

2. `Compressor`
- 负责压缩一组 `ContextItem`
- 例如：
  - `SlidingWindowCompressor`
  - `TokenBudgetCompressor`
  - `SelectiveCompressor`
- 另外，history 有单独的 `HistoryCompactor`

3. `Formatter`
- 负责把非 history 的上下文格式化成文本
- 例如：
  - `PlainFormatter`
  - `MarkdownFormatter`
  - `XMLFormatter`

4. `ContextBuilder / ContextManager`
- 负责预算分配、来源收集、压缩、最终拼装 `messages`

---

## 2. 最简用法

如果你只想让 Agent 在构建 prompt 时自动管理 history，可以这样用：

```python
from agent import BasicAgent
from core.llm import EasyLLM
from context import ContextManager

llm = EasyLLM()
context_manager = ContextManager(max_tokens=8000)

agent = BasicAgent(
    name="assistant",
    llm=llm,
    context_manager=context_manager,
    history_via_context_manager=True,
)

result = agent.invoke("继续刚才的问题")
```

这时：

- `ContextManager.build_messages()` 会在预算内构造最终输入
- 超预算时会优先压缩旧 history
- 如果发生压缩，`BasicAgent` 会把压缩后的 history 直接保存下来
- 最近几轮 history 会按原始顺序保留

---

## 3. 默认行为

当前默认行为如下：

1. history 不乱序
- 保留到最终 `messages` 里的 history 始终按旧到新顺序输出

2. 非 history 来源合并到 system
- `rag` / `memory` / 其他 source 会先被 formatter 格式化
- 最终拼进 system message

3. history 单独压缩
- history 不走普通 formatter
- history 由 `HistoryCompactor` 处理

4. 默认 history 压缩器是规则式的
- 默认实现是 `RuleBasedHistoryCompactor`
- 不调用大模型
- 会把较老轮次压成简短的历史摘要

5. 压缩后可直接复用
- 压缩器返回的是一条或多条可直接继续喂给模型的历史消息
- `BasicAgent` 在使用 `ContextManager` 时会保存这份压缩后的 history
- 完整原始过程继续保存在 `agent.trace_history`

---

## 4. 手动调用 `build_messages`

如果你不通过 Agent，也可以直接使用 `ContextManager`：

```python
from context import ContextManager

manager = ContextManager(max_tokens=4000)

history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你？"},
]

messages = manager.build_messages(
    query="继续刚才的问题",
    system_prompt="你是一个有帮助的助手",
    history=history,
    include_history=True,
    include_query=True,
)
```

返回结果大致类似：

```python
[
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你？"},
    {"role": "user", "content": "继续刚才的问题"},
]
```

---

## 5. 接入 RAG / Memory

你可以把其他来源注册到 `ContextManager`：

```python
from context import ContextManager
from context.source import RAGContextSource, MemoryContextSource

manager = ContextManager(max_tokens=8000)
manager.add_source(RAGContextSource(pipeline), weight=0.8)
manager.add_source(MemoryContextSource(memory_manage), weight=0.6)
```

这时最终输入会变成：

- system prompt
- 格式化后的 RAG / memory 上下文
- 压缩后的 history
- 当前 query

---

## 6. History 压缩结果

history 超预算时，`ContextBuilder` 会直接返回压缩后的 history 消息列表。

你可以通过：

```python
manager.last_compacted_history
manager.last_history_was_compacted
```

查看最近一次压缩结果。

如果你要手动先压缩 history，也可以直接调用：

```python
compacted_history = manager.compact_history(history, max_tokens=1200)
```

在 `BasicAgent` 中，如果本轮发生压缩，这份 `compacted_history` 会直接替换 `agent.history`，后续轮次继续使用它。

---

## 7. 自定义 Formatter

formatter 只负责把非 history 上下文化成文本，不负责预算和 history 压缩。

### 7.1 使用内置 formatter

```python
from context import ContextManager
from context.formatter import XMLFormatter

manager = ContextManager(max_tokens=8000)
manager.set_formatter(XMLFormatter())
```

### 7.2 自定义 formatter

```python
from context.formatter.base import BaseFormatter

class SimpleFormatter(BaseFormatter):
    def format(self, items, source: str = "") -> str:
        return "\n".join(item.content for item in items)

manager.set_formatter(SimpleFormatter())
```

---

## 8. 自定义普通 Compressor

普通 compressor 用于 source 级或全局压缩，不负责 history 语义。

### 8.1 给某个 source 设置压缩器

```python
from context.compressor import SlidingWindowCompressor

manager.add_source(
    RAGContextSource(pipeline),
    weight=0.8,
    compressor=SlidingWindowCompressor(max_items=5),
)
```

### 8.2 设置全局压缩器

```python
from context.compressor import TokenBudgetCompressor

manager.set_compressor(TokenBudgetCompressor(max_tokens=3000))
```

---

## 9. 自定义 HistoryCompactor

这是这次重构后新增的正式扩展点。

### 9.1 使用默认规则式压缩器

```python
from context import ContextManager
from context.compressor.history import RuleBasedHistoryCompactor

manager = ContextManager(max_tokens=8000)
manager.set_history_compactor(
    RuleBasedHistoryCompactor(recent_turns=4, min_recent_turns=1)
)
```

### 9.2 自定义 history 压缩器

你可以实现 `BaseHistoryCompactor`：

```python
from context.compressor.history import BaseHistoryCompactor

class KeepRecentOnlyCompactor(BaseHistoryCompactor):
    def compact(self, history, max_tokens):
        history = list(history or [])
        return history[-4:]

manager.set_history_compactor(KeepRecentOnlyCompactor())
```

如果你想接真正的 LLM summary，可以直接使用：

```python
from context.compressor.history import LLMHistoryCompactor

manager.set_history_compactor(
    LLMHistoryCompactor(llm=llm, max_summary_messages=3)
)
```

`LLMHistoryCompactor` 会把完整 history 压成 1 到数条 `user/assistant` 历史消息，而不是单独返回一个摘要字符串。

---

## 10. 高级用法：预配置 Builder

如果你想完全自己装配，可以直接创建 `ContextBuilder`：

```python
from context import ContextBuilder, ContextManager
from context.token import TokenBudget, TokenCounter
from context.formatter import MarkdownFormatter
from context.compressor import TokenBudgetCompressor
from context.compressor.history import RuleBasedHistoryCompactor

builder = ContextBuilder(
    budget=TokenBudget(max_tokens=8000),
    counter=TokenCounter(model="gpt-4"),
)

builder.set_formatter(MarkdownFormatter())
builder.set_compressor(TokenBudgetCompressor(max_tokens=5000))
builder.set_history_compactor(RuleBasedHistoryCompactor(recent_turns=6))

manager = ContextManager(builder=builder)
```

这种方式适合：

- 你想严格控制 token 预算
- 你想替换默认计数器
- 你想预装配一整套上下文策略

---

## 11. 推荐实践

推荐默认做法：

1. 普通业务先用 `ContextManager(max_tokens=...)`
2. 让 Agent 通过 `history_via_context_manager=True` 接入
3. 保留默认 `RuleBasedHistoryCompactor`
4. 只有在真的需要时，再自定义 formatter / compressor / history compactor

推荐你优先查看这几个状态：

1. `agent.history`
- 看当前继续对话真正会使用的 history

2. `manager.last_compacted_history`
- 看最近一次压缩后生成了哪些 history 消息

3. `agent.trace_history`
- 看完整执行轨迹

---

## 12. 示例程序

仓库里有一个完整的本地示例：

- `example/example_context_compaction.py`

运行：

```bash
/home/wxd/miniconda3/envs/llm/bin/python example/example_context_compaction.py
```

这个示例会输出：

- 压缩后保存在 `agent.history` 上的 history
- `manager.last_compacted_history`
- 最终 `build_messages()` 结果
- 最终 token 数和预算

适合直接观察：

- history 怎么被直接压成新的消息列表
- 压缩后为什么后续轮次不需要再传原始大 history
- 最终喂给模型的 messages 长什么样
