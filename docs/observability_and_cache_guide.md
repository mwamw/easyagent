# Observability And Cache Guide

Observability 模块负责记录 Agent、LLM、Tool 和 cache 相关运行信息。对框架使用者来说，这一层的意义不是“多一点日志”，而是让你能回答这些关键问题：

- 这次请求为什么慢？
- 这轮到底调用了几个工具？
- prompt token 为什么突然变多？
- cache 命中率为什么掉了？
- 是 system 变了，还是 history 被压缩了？

如果你把 EasyAgent 用来做真正的产品，这一层基本是必配，而不是可选装饰。

相关文档：

- [Prompt System Guide](./prompt_system_guide.md)
- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Context And Compaction Guide](./context_and_compaction_guide.md)

## 1. 核心对象

当前最重要的对象有：

- `BaseObservabilityRecorder`
  - 抽象接口。
- `InMemoryObservabilityRecorder`
  - 默认内存实现。
- `agent.get_context_usage()`
  - 从 Agent 侧查看当前上下文和 cache 状态的主要入口。

Recorder 主要会记录三类数据：

- agent runs
- llm requests
- tool executions

以及一类诊断数据：

- cache breaks

## 2. Recorder 记录哪些事件

`InMemoryObservabilityRecorder` 的事件面可以概括为：

### Agent 级

- `begin_agent_run`
- `end_agent_run`

记录：

- query
- mode
- stream 与否
- success / error
- duration

### LLM 级

- `begin_llm_request`
- `end_llm_request`

记录：

- provider
- model
- request kind
- tools enabled
- input/output/total tokens
- cached token 相关字段
- reasoning tokens
- usage source

### Tool 级

记录：

- tool name
- success / error
- duration
- 参数和结果摘要

### Cache break 级

记录：

- break reason
- changed fields
- layer 归因

## 3. 为什么 cache 指标不能直接看 provider 原始字段

因为不同 provider 的 usage 语义不一样。

### Anthropic 风格

常见字段：

- `inputTokens`
- `cacheReadTokens`
- `cacheCreationTokens`

这里 `inputTokens` 更像“未命中的输入部分”，而不是整个 prompt 总量。

### OpenAI / OpenAI-compatible 风格

常见字段：

- `prompt_tokens`
- `cached_tokens`

这里 `prompt_tokens` 通常表示整个 prompt 总量，`cached_tokens` 是其中命中的部分。

### Google 风格

有时能拿到 usage，但不一定有真实 cache usage 字段。

所以 EasyAgent 现在会把这些字段归一化成统一视图，而不是直接把原始字段拿来算比率。

## 4. 统一后的 cache 指标怎么理解

当前文档中最重要的几个指标是：

- `promptTokensTotal`
  - 统一语义下的 prompt 总量
- `promptTokensUncached`
  - 没有命中的部分
- `promptTokensCached`
  - 命中的部分
- `cacheHitRatioNormalized`
  - 标准化后的命中率
- `cacheUsageSemantics`
  - 当前使用的是哪种 provider 语义

推荐你始终优先看这些“normalized”指标，而不是直接比较各 provider 原始 usage。

## 5. `cacheUsageSemantics` 有什么用

它告诉你当前 usage 的归一化来源。

常见值包括：

- `anthropic_style`
- `openai_style`
- `google_style`
- `unknown`

实际意义是：

- 你知道当前 hit ratio 是按哪套逻辑算出来的
- 你知道某些 provider 为什么只能给出部分指标

## 6. `cache break` 是什么

很多产品只看“这一轮 cached tokens 是多少”，但这还不够。真正有用的是知道：

> 为什么这轮突然不命中了？

这就是 `cache break` 记录的作用。

当前常见 break 原因包括：

- `system_core_changed`
- `runtime_reminder_changed`
- `expanded_tool_set_changed`
- `message_prefix_changed`
- `skill_body_changed`
- `reasoning_changed`
- `provider_changed`
- `history_compacted`

## 7. `cache layer breaks` 是什么

除了 break reason，还会进一步按层归因：

- `system`
- `tools`
- `runtime_reminders`
- `messages`
- `skills`
- `provider`
- `reasoning`
- `other`

这个维度的价值是：你不只知道“断了”，还知道“是在哪一层断的”。

## 8. 什么会最常导致 cache 命中率下降

在产品里最常见的是这几类：

### 1. System prompt 抖动

例如把大量动态信息放进 `system_core`。

### 2. Tool schema 集合变化

例如一次性把很多低频工具全量暴露，或者 expanded tools 每轮都不同。

### 3. Runtime reminder 变化过于频繁

例如把本应属于动态尾部的信息放进稳定 reminder 前缀。

### 4. 历史压缩

这会直接改变 messages prefix。

### 5. Skill 正文切换

resident/on-demand skill 使用方式不合理时非常容易导致前缀变化。

## 9. Deferred tools 对 cache 的影响

Deferred tools 的核心价值之一就是稳定初始 tools payload。

如果你全量暴露所有工具 schema，会带来两个问题：

- prompt 很重
- tools hash 容易抖

Deferred 模式下：

- 常驻只暴露精简目录和少量默认工具
- 真正需要时再展开 schema

这通常会显著改善产品级 cache 命中。

## 10. Runtime reminders 对 cache 的影响

Runtime reminder 的设计目标不是“少发一段文本”，而是：

- 把高可信但动态的运行时信息从 system core 拆出去
- 每次请求 prepend 一次
- 不写入长期 canonical history

这样做的收益是：

- system 更稳定
- 历史不被 reminder 污染
- cache 结构更清晰

## 11. History compaction 对 cache 的影响

History compaction 通常会触发：

- `history_compacted`
- `message_prefix_changed`

所以你看到 compaction 后 cache 命中下降，通常是正常现象，不是 bug。

产品真正要做的是：

- 不要过早压缩
- 压缩后尽快形成新的稳定前缀

## 12. 一次典型诊断流程

下面是一条实用诊断路径：

1. 先看 `get_summary()`
2. 看：
   - `promptTokensTotal`
   - `promptTokensCached`
   - `cacheHitRatioNormalized`
3. 如果 hit ratio 异常下降，再看：
   - `cacheBreakReason`
   - `cacheLayerBreaks`
4. 再结合：
   - 最近有没有切 model/provider
   - 有没有 load/unload skill
   - 有没有 expanded tool set 改变
   - 有没有 history compaction

这是定位 cache 问题最快的方式。

## 13. `agent.get_context_usage()` 是看什么的

`get_context_usage()` 偏当前请求视角。

你通常会在这里看到：

- 当前上下文估算
- request frame 各层信息
- cache state
- 上次 signature / usage / break

它适合：

- 调试“这一轮为什么上下文突然大了”
- 看当前 request layering 是否符合预期

## 14. `get_summary()` 是看什么的

`get_summary()` 偏会话级统计。

它更适合：

- 看一个 session 内整体请求规模
- 看总工具调用次数
- 看一段工作流的平均时延
- 看整体 cache 命中情况

## 15. 如何接入 `BasicAgent`

最常见方式：

```python
from easyagent import BasicAgent, EasyLLM
from easyagent.observability import InMemoryObservabilityRecorder

recorder = InMemoryObservabilityRecorder(agent_name="code-agent")

agent = BasicAgent(
    name="code-agent",
    llm=EasyLLM(),
    observability_recorder=recorder,
)
```

然后在产品代码里：

```python
summary = recorder.get_summary()
usage = agent.get_context_usage()
```

## 16. 推荐的产品展示方式

如果你在做 UI 或 CLI，建议至少展示这些字段：

- 当前 provider / model
- 本轮 input / output tokens
- cached / uncached prompt tokens
- cache hit ratio
- 最近一次 cache break reason
- 工具调用次数
- 当前上下文估算

不要只显示一个 `total_tokens`，那对调试几乎没有帮助。

## 17. 常见坑

### 坑一：直接拿 provider 原始 usage 做跨 provider 比较

这会因为字段语义不同而误判。

### 坑二：把 cache hit ratio 当作唯一优化目标

命中率高不代表产品就好。你还要平衡：

- 输出质量
- 时延
- 工具可见性
- 技能可用性

### 坑三：不记录 cache break

只看命中率，不知道为什么掉，是无法真正优化的。

### 坑四：压缩后看到命中下降就认为系统坏了

压缩是有代价的，关键是要有明确 break reason 和稳定后的新前缀。

### 坑五：把 reminder、skill、expanded tools 的变化混在一起看

更好的做法是始终按 layer 分析。
