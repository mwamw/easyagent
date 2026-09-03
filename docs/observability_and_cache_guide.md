# Observability And Training Guide

Observability 是可选 Agent 模块。启用后，它订阅统一 `RuntimeEventBus`，自动记录 Agent 运行期间发生的 LLM、Tool、compaction、成功、失败和中断事件。

Training 不记录运行过程，只消费 Observability 已保存的数据。

## 1. 启用方式

内存存储：

```python
from easyagent import InMemoryObservabilityStore

agent.with_observability(store=InMemoryObservabilityStore())
```

SQLite：

```python
agent.with_observability(path=".easyagent/observability.sqlite3")
```

自定义 Manager 应继承 `BaseObservabilityManager`，实现事件绑定、事件消费、`list()` 和关闭；自定义 Store 应继承 `BaseObservabilityStore`。`TrainingExporter` 直接消费二者共有的 `list() -> list[AgentInvoke]`，不依赖默认实现。

## 2. LLMInvoke

一次 LLM 调用对应一条 `LLMInvoke`：

- `input`：完整 `list[CanonicalMessage]`
- `output`：完整 `list[CanonicalMessage]`
- `tools`：本次发送的工具 schema
- `options`：温度、reasoning 等请求选项
- `stats`：token、cache、reasoning、duration、success 和 error
- `metadata`：provider、round 等扩展信息

输入输出都使用 canonical history 数据结构，因此训练模块不需要理解 OpenAI、Anthropic 或 Google 的原始消息格式。

## 3. AgentInvoke

一次 `agent.invoke/ainvoke/stream/astream` 对应一条 `AgentInvoke`：

- `query`
- `llm_invokes`
- `trace`
- `output`
- `stats`
- `parent_invoke_id`
- `metadata`

Agent stats 聚合本次调用的 LLM 次数、工具次数、token、duration 和错误。`parent_invoke_id` 用于把父子 Agent 连接成 rollout。

## 4. 查询与标注

```python
latest = agent.observability.latest()
records = agent.observability.list()
summary = agent.observability.summary()

agent.observability.annotate(
    {"reward": 1.0, "reviewed": True},
    invoke_id=latest.invoke_id,
)
```

业务标签、人工评分和数据清洗标记应放入 metadata，不需要扩展核心 schema。

## 5. Cache 和 compaction

LLM stats 保留统一字段：

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cached_input_tokens`
- `cache_read_tokens`
- `cache_creation_tokens`
- `reasoning_tokens`
- `tool_use_prompt_tokens`

当 executor 执行 history compaction 时会发布 `history.compacted`。Observability 将其记录为 `history_compacted` cache break，并保留压缩前后 token 和 metadata。

`agent.get_context_usage()` 提供当前请求的轻量视图：

```python
{
    "estimatedRequestTokens": 1234,
    "canonicalMessages": 8,
    "replayMessages": 8,
    "provider": "openai",
}
```

Provider 返回的实际 usage 进入 LLMInvoke；当前请求估算不伪装成 provider usage。

## 6. 训练数据导出

```python
from easyagent import TrainingDataFormat, TrainingExporter

exporter = TrainingExporter.from_agent(agent)
report = exporter.export(
    ".easyagent/training",
    formats=[
        TrainingDataFormat.STEP_SFT,
        TrainingDataFormat.TRACE_SFT,
        TrainingDataFormat.AGENTIC_ROLLOUT,
    ],
)
```

输出是 JSONL：

- `step_sft`：每个成功 LLMInvoke 一条样本，包含完整 input、tools、options 和 output。
- `trace_sft`：每个成功 AgentInvoke 一条样本，包含 query、完整 trace 和最终 output。
- `agentic_rollout`：按 parent invoke 组织父子 Agent trajectory。

## 7. 过滤与清洗

默认 `SuccessfulAgentInvokeFilter` 只接受已结束且成功的 AgentInvoke。

```python
from easyagent import AgentInvoke, TrainingDataFilter


class ReviewedFilter(TrainingDataFilter):
    def accept(self, invoke: AgentInvoke) -> bool:
        return invoke.stats.success and invoke.metadata.get("reviewed") is True

    def transform(self, invoke: AgentInvoke) -> AgentInvoke:
        cleaned = invoke.model_copy(deep=True)
        cleaned.metadata.pop("private_note", None)
        return cleaned


report = exporter.export(
    ".easyagent/reviewed-training",
    data_filter=ReviewedFilter(),
)
```

过滤器只处理记录副本，不修改原始观测数据。

## 8. Session 行为

Observability 模块的状态可以进入 session snapshot。恢复标准 manager 时会恢复已完成记录；自定义 manager 需要显式传给 `load_session`。未启用 Observability 时，Agent 不创建 trace 数据库，TrainingExporter 会明确报错。
