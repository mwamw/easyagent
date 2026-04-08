# 流式输出使用说明

本文档说明 EasyAgent 当前的流式调用能力，包括普通流式输出和流式工具调用。

---

## 1. 支持范围

当前已支持：

- 无工具模式流式输出
- 工具模式流式输出
- 同步流式调用
- 异步流式调用

当前覆盖的 Provider：

- OpenAI / DeepSeek / Qwen / Kimi / Zhipu / Ollama / vLLM / ModelScope
- Google
- Anthropic
- OpenAI Responses API

说明：

- OpenAI-compatible Provider 复用统一的 Chat Completions 流式工具调用实现
- `openai_responses` 使用独立的 Responses API 流事件解析逻辑
- Anthropic / Google 在消息回填格式上做了额外适配

---

## 2. BasicAgent 流式入口

### 2.1 普通流式

同步：

```python
result = agent.stream_invoke("介绍一下 EasyAgent")
print(result)
```

异步：

```python
result = await agent.astream_invoke("介绍一下 EasyAgent")
print(result)
```

说明：

- 控制台会实时打印文本片段
- 返回值始终是最终完整文本

### 2.2 工具模式流式

当 `BasicAgent(enable_tool=True, tool_registry=...)` 时：

```python
result = agent.stream_invoke("帮我查天气并总结")
print(result)
```

```python
result = await agent.astream_invoke("帮我查天气并总结")
print(result)
```

说明：

- 工具模式下不再抛出 `NotImplementedError`
- Agent 会在流式生成过程中执行工具，并继续后续轮次生成

---

## 3. 事件流接口

如果你需要自己接管输出，而不是直接打印到控制台，可以使用事件流接口。

同步：

```python
for event in agent.stream_invoke_with_tool("帮我查询北京天气"):
    print(event)
```

异步：

```python
async for event in agent.astream_invoke_with_tool("帮我查询北京天气"):
    print(event)
```

事件类型如下。

### 3.1 `thinking_delta`

模型思考过程增量。只有开启 `verbose_thinking=True` 时，Agent 才会向外继续产出。

```python
{
    "type": "thinking_delta",
    "delta": "我需要先调用天气工具"
}
```

### 3.2 `text_delta`

最终回答正文的流式文本分片。

```python
{
    "type": "text_delta",
    "delta": "北京今天"
}
```

### 3.3 `tool_call`

Agent 已经拿到完整工具调用参数，准备执行工具。

```python
{
    "type": "tool_call",
    "tool_name": "weather",
    "tool_args": {"city": "Beijing"},
    "tool_id": "call_1"
}
```

### 3.4 `tool_result`

工具执行完成后的结果事件。

```python
{
    "type": "tool_result",
    "tool_name": "weather",
    "tool_id": "call_1",
    "tool_args": {"city": "Beijing"},
    "content": "晴，22 摄氏度"
}
```

### 3.5 `final`

完整最终答案。

```python
{
    "type": "final",
    "content": "北京今天晴，22 摄氏度，适合出行。",
    "thinking": ""
}
```

### 3.6 `error`

流式调用失败时产出。

```python
{
    "type": "error",
    "error": "..."
}
```

---

## 4. 推荐用法

如果你只是想“边打字边看到结果”，直接用：

- `stream_invoke()`
- `astream_invoke()`

如果你需要前端 UI、SSE、WebSocket 或自定义渲染，建议用：

- `stream_invoke_with_tool()`
- `astream_invoke_with_tool()`

原因是这两个接口会暴露结构化事件，便于你在界面上分别展示：

- 模型思考
- 普通文本
- 工具调用开始
- 工具执行结果
- 最终答案

---

## 5. 最小示例

```python
from pydantic import BaseModel

from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM
from Tool.ToolRegistry import ToolRegistry


class WeatherParams(BaseModel):
    city: str


registry = ToolRegistry()


@registry.tool("weather", "查询天气", WeatherParams)
def weather(city: str) -> str:
    return f"{city}：晴，22 摄氏度"


llm = EasyLLM(provider="openai")

agent = BasicAgent(
    name="assistant",
    llm=llm,
    enable_tool=True,
    tool_registry=registry,
    verbose_thinking=True,
)


async def main():
    async for event in agent.astream_invoke_with_tool("北京天气怎么样？"):
        print(event)
```

---

## 6. 行为说明

- 工具调用采用“LLM 流式生成 -> 聚合完整 tool call -> 执行工具 -> 继续下一轮 LLM”的循环
- `stream_invoke()` / `astream_invoke()` 会自动打印 `text_delta`，并返回最终结果
- `stream_invoke_with_tool()` / `astream_invoke_with_tool()` 不负责 UI，只负责产出事件
- 对话历史只在最终完成后写入 Agent history，避免中间半成品污染会话
- 工具执行仍然会触发现有 callback 生命周期

---

## 7. 注意事项

- 使用工具模式时必须提供 `ToolRegistry`
- 不同 Provider 的底层流事件格式不同，但 Agent 层看到的是统一事件格式
- `thinking_delta` 是否可用取决于底层模型和 Provider 是否提供 reasoning 流
- 如果某些 OpenAI-compatible 服务不返回标准的 `tool_calls delta`，流式工具能力可能受服务端兼容程度影响

