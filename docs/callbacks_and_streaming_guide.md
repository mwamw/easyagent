# Callbacks And Streaming Guide

Callback 系统是 EasyAgent 的**非阻塞观察层**。  
它用来做日志、流式 UI、指标采集、调试输出，而不是用来阻断执行。

相关文档：

- [Hooks And Guardrails Guide](./hooks_and_guardrails_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 1. Callback 系统解决什么问题

如果没有 callback，你通常只能：

- 打日志
- 改 agent 主流程
- 硬插调试代码

这会导致产品层和框架层耦合很重。  
Callback 的目标就是给你一个稳定的事件面。

## 2. 核心对象

### `CallbackEvent`

通用事件对象。主要字段：

- `event_type`
- `timestamp`
- `data`

### `BaseCallback`

所有 callback 的基类。  
你可以选择性覆写感兴趣的事件。

### `CallbackManager`

统一管理多个 callback，并把事件分发给它们。

### 现成 callback

- `StreamingCallback`
- `LoggingCallback`
- `MetricsCallback`

## 3. 主要事件点

### Agent 级

- `on_agent_start`
- `on_agent_end`

### LLM 级

- `on_llm_start`
- `on_llm_end`

### Tool 级

- `on_tool_start`
- `on_tool_end`

### Chain / Thinking 级

- `on_chain_start`
- `on_chain_end`

### 错误

- `on_error`

## 4. 什么适合用 Callback 做

### 流式 UI

把 token / thinking / tool 状态实时刷到：

- CLI
- Web UI
- IDE 面板

### 日志

记录：

- 调了哪些工具
- 哪一轮触发了中断
- 原始 provider 响应摘要

### 指标

记录：

- 耗时
- token
- tool 次数
- 错误数

## 5. 什么不适合用 Callback 做

以下更适合 Hook 或 Permission：

- 阻止某次请求
- 改写请求 payload
- 拒绝某个工具

Callback 更像“旁观者”，不是“裁判”。

## 6. 最小自定义示例

```python
from easyagent.callbacks import BaseCallback

class PrintToolCallback(BaseCallback):
    def on_tool_start(self, tool_name, tool_input, **kwargs):
        print("tool start:", tool_name, tool_input)

    def on_tool_end(self, tool_name, tool_output, success=True, error=None, **kwargs):
        print("tool end:", tool_name, success)
```

## 7. 和 Agent 的集成

```python
from easyagent.callbacks import CallbackManager

callback_manager = CallbackManager([PrintToolCallback()])
agent = BasicAgent(...).with_callbacks(callback_manager)
```

## 8. 产品集成建议

### CLI

优先用 `StreamingCallback` 做逐步渲染。

### Web / IDE

建议把 callback 事件转换成你自己的事件总线格式，再分发给前端。

### 监控系统

把 callback 和 observability recorder 结合使用：

- callback 负责实时流
- recorder 负责会后汇总

## 9. 常见坑

### 在 callback 里写阻断逻辑

这会让职责混乱。  
阻断应放到 hook / permission。

### 在 callback 里做过重 I/O

callback 在高频事件下可能非常频繁，过重的 I/O 会拖慢整条链路。
