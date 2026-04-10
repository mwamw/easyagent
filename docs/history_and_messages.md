# History 与消息回填说明

本文档说明 EasyAgent 当前在不同 Provider 下如何处理：

- `history` 中保存什么
- `thinking` 保存到哪里
- 工具调用前后的消息如何回填
- 下一轮喂给模型时哪些字段会继续保留

本文档的例子尽量保持“原汁原味”，贴近框架当前真实的 `history` 结构，而不是人为简化后的伪格式。

---

## 1. 总体规则

当前框架已经统一了下面几条规则：

- `history` 保存正式会话消息与工具轨迹
- `thinking` 默认进入 `trace_history`
- 工具轮次会把中间 assistant/tool 轨迹一并写回 `history`
- `openai_responses` 不再把 `reasoning` item 直接回填进正式 `history`
- 喂给下一轮模型时，会裁掉纯噪音字段，避免 `id` / `status` / `encrypted_content` 之类无意义元数据污染上下文

`trace_history` 当前按“完整会话转录”设计：

- 不保存 `system prompt`
- 不保存发给 LLM 的完整 `messages`
- 不保存流式 `text_delta`
- 只保存用户原始输入、聚合后的 reasoning、assistant 消息、tool_call、tool_result 与 turn 结束信息

可以把当前数据分成两层：

- 正式会话层：`agent.history`
- 会话转录层：`agent.trace_history`

---

## 2. `history` 里会出现什么

`history` 现在不是“只有 `Message` 对象”的列表，而是：

- `UserMessage`
- `AssistantMessage`
- Provider-specific dict
- 工具结果 dict

也就是说，一个真实的 `history` 可能长这样：

```python
[
    UserMessage(...),
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    AssistantMessage(...),
]
```

或者在 `openai_responses` 下长这样：

```python
[
    UserMessage(...),
    {"type": "message", "role": "assistant", "content": [...]},
    {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."},
    {"type": "function_call_output", "call_id": "...", "output": "..."},
    AssistantMessage(...),
]
```

---

## 3. Thinking 到哪里去了

当前实现里，`thinking` 不会单独写成一条正式 history 消息。

它的去向是：

- 非流式：通过 `llm.get_thinking_content(...)` 提取，写入 `agent.trace_history`
- 流式：通过 `thinking_delta` 事件增量输出，并在 round 结束后聚合写入 `agent.trace_history`

所以：

- 你在控制台上能看到 `thinking content:`
- 但 `agent.history` 里通常不会多出一条 `"type": "reasoning"` 或 `"role": "assistant"` 的 thinking 消息
- 如果你想看完整输入输出、工具执行和每轮推进过程，应该优先看 `agent.get_trace_history()`

这是一个有意的分层：

- 正式 history 用于会话续写
- trace 用于会话转录、回放和会话恢复

### 3.1 Trace 的真实结构

`trace_history` 现在保存的是完整会话转录。一个工具调用轮次结束后，大致会看到：

```python
[
    {
        "id": "evt_000001",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 1,
        "type": "user_message",
        "timestamp": "2026-04-09T20:00:00.000000",
        "role": "user",
        "content": "帮我计算 3^22",
        "metadata": {},
    },
    {
        "id": "evt_000002",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 2,
        "type": "reasoning",
        "timestamp": "2026-04-09T20:00:01.000000",
        "role": "assistant",
        "content": "我先调用计算工具。",
        "round": 1,
        "metadata": {"mode": "tool", "stream": True, "visibility": "internal"},
    },
    {
        "id": "evt_000003",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 3,
        "type": "assistant_message",
        "timestamp": "2026-04-09T20:00:01.000000",
        "role": "assistant",
        "content": "我先调用计算工具。",
        "parent_id": "evt_000002",
        "round": 1,
        "metadata": {"stage": "pre_tool", "mode": "tool", "stream": True},
    },
    {
        "id": "evt_000004",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 4,
        "type": "tool_call",
        "timestamp": "2026-04-09T20:00:01.000000",
        "role": "assistant",
        "content": "",
        "parent_id": "evt_000003",
        "round": 1,
        "tool_name": "calculator",
        "tool_args": {"expression": "3**22"},
        "tool_call_id": "call_1",
        "metadata": {"mode": "tool", "stream": True},
    },
    {
        "id": "evt_000005",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 5,
        "type": "tool_result",
        "timestamp": "2026-04-09T20:00:01.200000",
        "role": "tool",
        "parent_id": "evt_000004",
        "round": 1,
        "tool_name": "calculator",
        "tool_args": {"expression": "3**22"},
        "tool_call_id": "call_1",
        "content": "31381059609",
        "metadata": {"mode": "tool", "stream": True, "success": True},
    },
    {
        "id": "evt_000006",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 6,
        "type": "assistant_message",
        "timestamp": "2026-04-09T20:00:02.000000",
        "role": "assistant",
        "round": 2,
        "content": "3^22 = 31381059609",
        "parent_id": "evt_000005",
        "metadata": {"stage": "final", "mode": "tool", "stream": True},
    },
    {
        "id": "evt_000007",
        "session_id": "trace_xxx",
        "turn_id": "turn_0001",
        "seq": 7,
        "type": "turn_end",
        "timestamp": "2026-04-09T20:00:02.000100",
        "role": "assistant",
        "content": "",
        "metadata": {"mode": "tool", "stream": True, "status": "completed", "final_event_id": "evt_000006"},
    },
]
```

如果你只关心推理文本，可以直接从 `trace_history` 中筛出 `type == "reasoning"` 的事件，例如：

```python
[
    event["content"]
    for event in agent.get_trace_history()
    if event["type"] == "reasoning"
]
```

也就是说：

- `trace_history` 适合回放、会话恢复和可视化
- reasoning 文本只是 trace 中的一类事件

---

## 4. OpenAI-compatible Provider 的真实例子

这里的 “OpenAI-compatible” 包括：

- OpenAI
- DeepSeek
- Qwen
- Kimi
- Zhipu
- Ollama
- vLLM
- ModelScope
- Google 的 OpenAI-compatible 路径

### 4.1 无工具调用

如果模型只是普通回复，`history` 大致如下：

```python
[
    UserMessage(
        role="user",
        content="介绍一下你自己",
        time=datetime.datetime(...),
        metadata={},
    ),
    AssistantMessage(
        role="assistant",
        content="我是 EasyAgent 中的一个基础智能体实例。",
        time=datetime.datetime(...),
        metadata={},
    ),
]
```

如果底层模型带有 `reasoning_content`，thinking 会进入 `agent.trace_history`，以 `type == "reasoning"` 的事件形式保存。

### 4.2 工具调用

当模型返回一条 assistant message，其中同时包含正文和 `tool_calls` 时，保存下来的中间 history 类似这样：

```python
[
    UserMessage(
        role="user",
        content="帮我计算 3^22",
        time=datetime.datetime(...),
        metadata={},
    ),
    {
        "role": "assistant",
        "content": "我先调用计算工具。",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": "{\"expression\":\"3**22\"}",
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": "31381059609",
        "tool_call_id": "call_1",
    },
    AssistantMessage(
        role="assistant",
        content="3^22 = 31381059609",
        time=datetime.datetime(...),
        metadata={},
    ),
]
```

要点：

- 工具调用前的说明文字和 `tool_calls` 在同一条 assistant 消息里
- 工具结果是一条单独的 tool 消息
- 最后的正式回答仍然是 `AssistantMessage`

---

## 5. Anthropic 的真实例子

Anthropic 当前会保留 Claude 风格的 block 结构。工具调用前如果同时有文本和工具使用，`history` 会更接近下面这样：

```python
[
    UserMessage(
        role="user",
        content="查询北京天气并告诉我结果",
        time=datetime.datetime(...),
        metadata={},
    ),
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "我先查询天气。",
            },
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "weather",
                "input": {
                    "city": "Beijing",
                },
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "晴，22 摄氏度",
            }
        ],
    },
    AssistantMessage(
        role="assistant",
        content="北京今天晴，22 摄氏度。",
        time=datetime.datetime(...),
        metadata={},
    ),
]
```

要点：

- `tool_use` 在 assistant 的 block 列表里
- 工具结果不是 OpenAI 风格 `role=tool`，而是 Claude 风格 `role=user + tool_result block`
- 现在如果 Claude 同时给了正文和工具调用，正文不会再被丢掉

---

## 6. Google 的真实例子

Google 当前沿用 OpenAI-compatible 的大部分结构，但工具结果格式有差异：

```python
[
    UserMessage(
        role="user",
        content="帮我查询北京天气",
        time=datetime.datetime(...),
        metadata={},
    ),
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": "{\"city\":\"Beijing\"}",
                },
            }
        ],
    },
    {
        "role": "function",
        "content": "晴，22 摄氏度",
        "tool_call_id": "call_1",
        "name": "weather",
    },
    AssistantMessage(
        role="assistant",
        content="北京今天晴，22 摄氏度。",
        time=datetime.datetime(...),
        metadata={},
    ),
]
```

要点：

- 中间 assistant 消息仍然是 OpenAI-compatible 风格
- 但工具结果是 `role="function"`，并保留 `name`

---

## 7. OpenAI Responses 的真实例子

这是当前最复杂、也最容易出现误解的一类。

### 7.1 一轮里可能同时返回什么

`openai_responses` 一轮原始 output 里常见同时出现：

- `reasoning`
- `message`
- `function_call`

比如原始返回可能概念上类似：

```python
response.output == [
    ResponseReasoningItem(...),
    ResponseOutputMessage(..., phase="commentary"),
    ResponseFunctionToolCall(...),
]
```

### 7.2 现在写入 history 后是什么样

正式 history 不会直接保存 reasoning item，而是保留 message / function_call / function_call_output：

```python
[
    UserMessage(
        role="user",
        content="使用工具翻译下面的文字到英语并判断这个工具正确吗:\n你是谁，在哪里 \n 并帮我计算3^22",
        time=datetime.datetime(...),
        metadata={},
    ),
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "annotations": [],
                "text": "我先并行调用翻译和计算工具来获得准确结果。",
                "type": "output_text",
                "logprobs": [],
            }
        ],
        "phase": "commentary",
    },
    {
        "type": "function_call",
        "call_id": "call_btoDECL2aXd1xnSHfVBmU18t",
        "name": "translate_tool",
        "arguments": "{\"target_lang\":\"en\",\"text\":\"你是谁，在哪里\"}",
    },
    {
        "type": "function_call",
        "call_id": "call_t4kfWcK0ZCYj1vpRHCBuYf9S",
        "name": "calculator",
        "arguments": "{\"expression\":\"3**22\"}",
    },
    {
        "type": "function_call_output",
        "call_id": "call_btoDECL2aXd1xnSHfVBmU18t",
        "output": "Translated: 你是谁，在哪里",
    },
    {
        "type": "function_call_output",
        "call_id": "call_t4kfWcK0ZCYj1vpRHCBuYf9S",
        "output": "31381059609",
    },
    AssistantMessage(
        role="assistant",
        content="翻译结果：\n- 原文：你是谁，在哪里\n- 英语：The tool did not return a valid English translation.\n\n判断这个翻译工具这次是否正确：\n- 不正确。它返回的是 `Translated: 你是谁，在哪里`，实际仍然是中文，没有完成英文翻译。\n\n计算结果：\n- 3^22 = 31381059609\n\n如果你愿意，我也可以直接给出这句话更自然的英文翻译。",
        time=datetime.datetime(...),
        metadata={},
    ),
]
```

### 7.3 为什么没有 reasoning item

因为当前规则是：

- reasoning 保留到 `trace_history`
- 不作为正式 history 条目保存
- 也不再继续喂回下一轮模型

这样做是为了避免：

- `encrypted_content`
- 空的 `summary`
- 无意义 `id/status`

这类 provider 内部字段进入上下文，造成噪音。

### 7.4 为什么 `message` 还要保留

`message` 不是 reasoning，它通常是模型对用户可见的 commentary 文本，比如：

- “我先调用翻译工具”
- “我先并行调用两个工具”

这种内容是正式对话的一部分，因此会保留到 history。

### 7.5 下一轮喂给模型时会保留什么

当前会保留有语义价值的字段，例如：

```python
{
    "type": "message",
    "role": "assistant",
    "content": [...],
}
```

```python
{
    "type": "function_call",
    "call_id": "...",
    "name": "...",
    "arguments": "...",
}
```

```python
{
    "type": "function_call_output",
    "call_id": "...",
    "output": "...",
}
```

不会再继续喂：

- `id`
- `status`
- `encrypted_content`
- `reasoning`

---

## 8. 为什么 `history` 和 `trace_history` 要分开

如果把所有 thinking 都混进正式 history，会有几个问题：

- reasoning 往往不是面向用户的正式消息
- provider-specific 内部字段会污染上下文
- 下一轮模型会看到很多无意义的 trace 数据
- 会话持久化会越来越脏

所以当前设计是：

- `history` 用于会话续写
- `trace_history` 用于会话转录、回放和会话恢复

这是一个语义分层，不是信息丢失。

---

## 9. 当前行为总结

### 9.1 正式 history 中通常会保留

- 用户输入
- 工具调用前的正式 assistant message
- function_call / tool_use
- function_call_output / tool_result
- 最终 assistant answer

### 9.2 正式 history 中通常不会保留

- OpenAI Responses 的 reasoning item
- 纯 provider 内部元数据
- `encrypted_content`
- 无语义价值的 `id/status`

### 9.3 转录层会保留

- `trace_history`
- reasoning 事件
- assistant/tool 事件链
- turn 结束信息

---

## 10. 推荐查看方式

如果你要确认当前某个 Provider 的真实行为，推荐按下面顺序看：

1. 看控制台流式输出
2. 看 `agent.get_trace_history()`
3. 看 `agent.get_history()`
4. 如果还不够，再打开 DEBUG/TRACE 日志看 provider 原始返回

示例脚本：

- `example/example_stream_display.py`
- `example/example_stream_history.py`
