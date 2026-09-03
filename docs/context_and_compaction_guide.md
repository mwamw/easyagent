# Context And Compaction Guide

Context 模块负责“本次请求除了用户 query 之外，还应该带什么上下文”；Compaction 模块负责“历史太长时如何压缩，既不丢关键信息，也不把请求打爆”。

如果你在做的是一个真正的产品级 Agent，这一层通常比单纯的 prompt engineering 更重要，因为它决定了：

- 历史如何重放
- memory 如何进入请求
- RAG 如何进入请求
- 运行时提醒如何注入
- 上下文预算如何控制
- 历史压缩在什么时候发生

相关文档：

- [Prompt System Guide](./prompt_system_guide.md)
- [Prompt System Guide](./prompt_system_guide.md)
- [Memory System Guide](./memory_system_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 1. 核心对象

这一层最常见的对象有：

- `ContextManager`
  - 运行时总控，负责收集 source、构建请求输入、触发压缩。
- `ContextBuilder`
  - 把多个上下文来源和 formatter/compressor 编排成最终上下文。
- `BaseContextSource`
  - 单个上下文来源的抽象基类。
- `ContextItem`
  - 单个上下文片段的统一表示。
- `ContextWindow`
  - 一个上下文窗口，包含 items 和预算相关信息。
- `TokenBudget`
  - 预算对象，描述本次请求最大可用 token 空间。
- `TokenCounter`
  - 本地 token 估算器。
- `BaseHistoryCompactor`
  - 历史压缩器抽象接口。
- `RuleBasedHistoryCompactor`
  - 默认规则式压缩器。
- `LLMHistoryCompactor`
  - 借助模型做历史摘要与压缩的实现。
- `ReplayRequestInput`
  - 最终编译给 provider 的请求缓冲对象。

## 2. Context 层到底解决什么问题

很多项目一开始会把“上下文”理解成：

- 只拼接聊天历史
- 只在 system prompt 里塞点记忆

这在简单 demo 中还能工作，但在真正的产品里很快会失控。原因是请求内容来源很多：

- 用户当前 query
- canonical history
- replay history
- system reminder blocks
- memory
- mailbox
- RAG 检索结果
- 当前 active skills
- hook 产生的动态尾部

Context 层的职责，就是把这些来源变成一个有顺序、有预算、有压缩策略的请求。

## 3. `ContextManager` 负责什么

`ContextManager` 是对外最重要的上下文总控。

它主要负责：

1. 持有 `ContextBuilder`
2. 注册 `ContextSource`
3. 设置 formatter / compressor / history compactor
4. 根据本次请求组装 `ReplayRequestInput`
5. 在预算不够时触发 persistent history compaction

从使用角度看，你通常只需要做三件事：

```python
context_manager = ContextManager(max_tokens=12000)
context_manager.add_source(my_source)
context_manager.set_history_compactor(my_compactor)
```

然后把它交给 `BasicAgent`。

## 4. `ContextBuilder` 做什么

`ContextBuilder` 更偏“拼装器”，比 `ContextManager` 更底层。

你可以把职责理解成：

- `ContextManager`
  - 控制流程、预算和压缩时机
- `ContextBuilder`
  - 把具体 source 收集到一起，形成 request input

它适合自定义的情况包括：

- 你想改变 source 的收集顺序
- 你想改变 context item 的格式化方式
- 你想让不同 source 采用不同压缩策略

## 5. `BaseContextSource` 是什么

Context source 是“上下文来源”的抽象。每个来源只负责回答一个问题：

> 针对当前 Agent 和当前 query，你能提供哪些额外上下文？

常见 source 包括：

- memory source
- RAG source
- skill source
- 产品侧的 external state source
- 当前工作流状态 source

自定义一个 source 的典型方式是：

```python
from context.source.base import BaseContextSource


class TicketContextSource(BaseContextSource):
    def collect(self, query: str, **kwargs):
        ...
```

它比直接把内容硬塞进 prompt 更好的原因是：

- 能参与统一预算控制
- 能单独压缩
- 能统一格式化

## 6. 一次完整请求是怎么经过 Context 层的

这里用一条典型请求说明数据流：

1. 用户发起 `agent.invoke("检查缓存架构问题")`
2. Agent 当前已有：
   - canonical history
   - replay history
   - system reminder blocks
   - memory manager
   - skill manager
   - 若干 context sources
3. `ContextManager.build_request_input(...)` 被调用
4. `ContextBuilder` 从所有 source 收集 `ContextItem`
5. formatter 把这些 item 变成消息块或动态尾部
6. 构建 `ReplayRequestInput`
7. 如果预算不足：
   - `compact_persistent_history(...)` 被调用
8. 得到最终 request frame：
   - `system_prompt_blocks`
   - `system_reminder_blocks`
   - `replay_history`
   - `dynamic_tail_blocks`
   - `on_demand_expansion_blocks`

所以 Context 层不是历史模块的附属品，而是整个请求编排的关键。

## 7. canonical history、replay history、request-time reminders 的区别

这是最容易混淆的部分。

### canonical history

语义上的长期会话历史。

特点：

- 用来表达真正的对话内容
- 应尽量稳定、可持久化
- 不应该混入 provider 特定签名和 system reminder

### replay history

当前 provider/tool loop 用来重放的历史。

特点：

- 更接近真实 API request 的消息序列
- 允许包含 provider 相关结构
- 会随着 compaction、tool calls、streaming 回放逻辑变化

### request-time reminders

每次请求临时 prepend 的运行时提示。

特点：

- 不是长期历史
- 每次请求最多 prepend 一次
- 不写回 canonical history

推荐理解：

- canonical history 是“用户真正说过什么”
- replay history 是“本轮请求要怎么重放过去”
- reminders 是“框架在这次请求前额外提醒模型什么”

## 8. `TokenBudget` 和 `TokenCounter`

### `TokenBudget`

负责描述“这次请求最多能带多少上下文”。

它通常与：

- 模型上下文窗口
- 预留输出空间
- 压缩阈值

一起考虑。

### `TokenCounter`

负责本地估算 token。

要注意：

- 这是估算，不是 provider 账单真值
- 它主要用于预算决策和 compaction 触发
- 真正的 usage 以 provider 返回为准

## 9. History Compactor 的职责

当历史太长时，不能简单“丢掉最前面的几条”，否则容易破坏推理链或工具链。

History compactor 的目标是：

- 尽量保留最近几轮原文
- 用摘要压缩较老历史
- 尽量保持时间顺序
- 工具调用链尽量按轮次保留

## 10. `RuleBasedHistoryCompactor`

这是默认的低成本压缩器。

它的特点是：

- 不依赖额外 LLM 请求
- 稳定、快速、成本低
- 更适合作为兜底和回退路径

当前实现更偏“压缩工具结果和旧历史内容”，而不是做高质量抽象总结。

适合：

- 开发阶段
- 低成本产品
- provider 压缩失败时的 fallback

## 11. `LLMHistoryCompactor`

这是更强的压缩器，会借助模型生成摘要。

它的优点：

- 摘要质量通常更高
- 更擅长保留语义层信息

代价：

- 需要额外 LLM 请求
- 有失败和偏差的可能
- 会影响时延和成本

所以常见做法是：

- 优先 LLM compactor
- 失败时回退到规则压缩器

## 12. 历史压缩何时触发

一般在下面这些场景会考虑压缩：

1. replay history 的估算 token 已经接近预算上限
2. 当前模型上下文较小
3. 你显式强制压缩
4. 长对话 session 恢复后第一轮需要瘦身

当前设计里，压缩是由 `ContextManager.compact_persistent_history(...)` 控制的，而不是由某个 provider transport 私自决定。

## 13. 压缩与 cache 的关系

这也是产品里经常看错的地方。

压缩历史几乎一定会影响 cache，因为它会改变：

- message prefix
- replay history
- 请求签名

所以压缩后的常见现象是：

- `lastCacheBreak.reason = history_compacted`
- 命中率下降

这不是 bug，而是正常现象。正确目标不是“压缩后命中率不变”，而是：

- 只在确实需要时压缩
- 压缩后保持新的稳定前缀

## 14. 如何把 Context 模块接入 `BasicAgent`

最小方式：

```python
from easyagent import BasicAgent, EasyLLM
from easyagent.context import ContextManager

llm = EasyLLM()
context_manager = ContextManager(max_tokens=12000)

agent = BasicAgent(
    name="assistant",
    llm=llm,
).with_context(context_manager)
```

更典型的产品写法是：

```python
context_manager = ContextManager(max_tokens=12000)
context_manager.add_source(memory_source)
context_manager.add_source(rag_source)
context_manager.set_history_compactor(my_compactor)
```

## 15. 什么时候应该自定义 Context 层

以下场景建议自定义：

- 你需要把 memory、RAG、workflow state 混合排序
- 你希望某些 source 先于另一些 source 进入请求
- 你有自己的 token budget 策略
- 你需要更激进或更保守的压缩规则
- 你在做多产品框架，希望不同 Agent 共享不同 context recipe

## 16. 推荐实践

### 让 source 做“数据来源”，不要做“最终拼装”

source 负责提供材料，request frame 的最终布局仍然由 request/compiler 层决定。

### 让 reminder 和 history 分离

不要把 request-time reminder 写进 canonical history。

### 压缩器只在必要时触发

不是每轮都压缩。否则会既浪费请求，又让 cache 命中率持续变差。

### memory/RAG 优先做成 context source

而不是直接把结果手工塞进 system prompt。

## 17. 常见坑

### 坑一：把所有上下文都塞进 system prompt

这样会让：

- system hash 频繁变化
- cache 命中率差
- 结构难以维护

### 坑二：把 reminder 当成历史消息持久化

reminder 应该是 request-time 注入，不应污染长期对话。

### 坑三：把压缩当成“删消息”

压缩不是简单截断，而是重构旧历史的表达方式。

### 坑四：完全相信本地 token 估算

本地估算主要用于决策，provider usage 才是账单真值。

### 坑五：一个 source 既查数据又决定排版

这会让 source 变得难复用。更好的边界是：

- source 提供 item
- formatter 决定呈现
- compiler 决定进入哪一层
