# Phase I：Observability Metrics + Trace Summary

## 本阶段完成了什么

这一阶段把 EasyAgent 的观测层从“零散 callback 指标”补成了正式 runtime 能力。

当前已经落地：

- `observability/recorder.py`
  - `BaseObservabilityRecorder`
  - `InMemoryObservabilityRecorder`
- `BaseAgent` 内建观测接口
  - `get_observability_summary()`
  - `get_recent_observability_events()`
  - `get_trace_summary()`
  - `clear_observability()`
- plain / tool 两条主执行链都接入了观测
  - `agent/components/invocation_runner.py`
  - `agent/components/tool_loop_engine.py`
- session save/load 会保存与恢复 `observability_state`
- SDK 已新增 `easyagent.observability`

## 框架发生了什么变化

之前 EasyAgent 的运行信息主要散落在：

- `callback_manager`
- `trace_history`
- 各子系统自己的日志

这些信息都能看，但不形成统一口径。  
现在框架有了正式的观测层，`agent run / llm request / tool execution` 三类事件会被统一收集、汇总和恢复。

这意味着上层产品不需要再自己拼：

- 一共调了多少次 LLM
- 哪类请求最耗 token
- 哪个工具出错最多
- 最近几轮 trace 的 token / tool / duration 摘要

这些都可以直接从 agent 上拿到。

## 现阶段提供的能力

### 1. Agent Run 级观测

每次 `invoke / stream_invoke / ainvoke / astream_invoke` 都会记录：

- query
- mode
- stream
- duration
- success / error
- turn_id

### 2. LLM Request 级观测

每轮 LLM 请求会记录：

- request kind
- stream / tools enabled
- provider / model
- input / output / total tokens
- cost
- usage source
- duration
- error type

### 3. Tool Execution 级观测

每次工具执行会记录：

- tool name
- tool args
- status
- duration
- error type
- `side_effect_level / visibility_scope / resource_scope / toolSource`

### 4. 聚合视图

当前可直接读取：

- `agent.get_observability_summary()`
- `agent.get_recent_observability_events()`
- `agent.get_trace_summary()`

其中 summary 会聚合：

- agent runs
- llm requests
- tool calls
- token totals
- estimated cost
- request kind 分布
- tools used 分布
- error types
- 平均 duration

## 一个具体过程例子

假设你用 EasyAgent 做一个 code agent 管理器，真实执行过程可能是：

1. 主 agent 先做一次普通 `invoke()`，读取用户任务并生成计划
2. 再做一次 tool 模式 `invoke()`，调用 `FileRead / Grep / Agent / SendMessage`
3. 中途保存 session
4. 恢复后继续跑下一轮
5. 最后读取观测摘要，看这次任务一共消耗了多少 token、LLM 请求分布如何、哪些工具被频繁调用

现在这些信息不需要你在产品层另写埋点。  
`BaseAgent` 自己就能给出统一摘要和按 turn 的 trace summary。

## 真实使用方式

示例文件见：

- `example/example_phasei_observability_metrics.py`

这个 example 使用真实的：

```python
EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
```

它演示：

- 普通调用
- 工具调用
- stream 调用
- session save/load 后保留观测状态
- 输出 summary / recent events / trace summary

## 当前边界

这一阶段完成的是“最小但正式可用”的 observability 主线，不是最终所有观测形态的终点。

已经完成：

- runtime 内聚合
- session restore
- trace summary
- SDK 暴露

后续仍可继续增强：

- benchmark exporter
- 外部 metrics sink
- 更细的多 agent / MCP / codeintel 分桶统计
- 持久化 trace store

## 相关文件

- `observability/recorder.py`
- `core/agent.py`
- `agent/components/invocation_runner.py`
- `agent/components/tool_loop_engine.py`
- `easyagent/observability.py`
- `test/test_observability.py`
