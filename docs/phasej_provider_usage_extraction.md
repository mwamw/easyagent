# Phase J: Provider Usage Extraction

## 本次完成范围

这一轮没有扩新的 provider，而是把框架里已经存在的 4 条主线路径补成“优先读取真实 provider usage，而不是事后估算”：

- `openai` (`chat.completions`)
- `openai_responses` (`responses.create`)
- `anthropic_native` (`messages.create / messages.stream`)
- `google_native` (`models.generate_content / generate_content_stream`)

## 框架发生了什么变化

之前框架里的 `EasyLLM.extract_usage_metrics()` 是统一 best-effort 解析：

- 能从 response 里读到 `usage` 就读
- 读不到就退回本地估算

这有两个实际问题：

1. `openai chat` 非流式 transport 只返回 `choices[0].message`，原始 response 上的 `usage` 已经丢了。
2. 流式路径里 codec 即便拿到了最终 usage，`plain stream` 入口又会重新合成一个只包含 `content/thinking` 的 dict，把 usage 再丢一次。

现在这条链改成了：

1. `BaseProvider.get_usage_from_response()` 成为正式接口
2. `EasyLLM.extract_usage_metrics()` 先走 provider 正式实现，再走旧 fallback
3. `openai chat` transport 非流式保留原始 usage
4. 4 个 codec 的流式最终事件都带 usage
5. `BasicAgent.stream_invoke()` / `astream_invoke()` 不再把最终 usage 丢掉
6. observability 会把真实的 usage 细分字段记到事件和 summary 里

## 具体实现

### 1. Provider 正式接口

现在 `BaseProvider` 有了统一的：

- `get_usage_from_response(response)`

这不是一个估算接口，而是“当前 provider 如何从自己的原始返回值里拿真实 usage”的正式入口。

本次已经实现的字段标准化：

- `inputTokens`
- `outputTokens`
- `totalTokens`
- `cachedInputTokens`
- `reasoningTokens`
- `cacheReadTokens`
- `cacheCreationTokens`
- `toolUsePromptTokens`
- `costUsd`
- `usageSource`

### 2. OpenAI Chat

`openai` 现在修了两个关键点：

- 非流式 `invoke_raw()` 不再只返回裸 `message`，而是把原始 response 的 `usage` 一并保留下来
- 流式请求会自动加 `stream_options={"include_usage": true}`

这意味着：

- `llm.invoke_raw(...)` 后的 `llm.extract_usage_metrics(response)` 可以拿到真实 `prompt/completion/total`
- `llm.stream_events(...)` 的最终 `final_response` 事件也带 usage

### 3. OpenAI Responses

`openai_responses` 现在走 provider 专有解析：

- 读 `response.usage.input_tokens / output_tokens / total_tokens`
- 读 `input_tokens_details.cached_tokens`
- 读 `output_tokens_details.reasoning_tokens`

流式 `response.completed` 事件里的 usage 也会进最终事件。

### 4. Anthropic Native

`anthropic_native` 现在会：

- 非流式读取 `usage.input_tokens / output_tokens`
- 额外读取 `cache_read_input_tokens`
- 额外读取 `cache_creation_input_tokens`

流式路径会从 `message_start` / `message_delta` 累积 usage，并附在最终 `tool_calls` 或 `final_response` 事件上。

### 5. Google Native

`google_native` 现在会读取：

- `prompt_token_count`
- `candidates_token_count`
- `total_token_count`
- `cached_content_token_count`
- `thoughts_token_count`
- `tool_use_prompt_token_count`

流式路径会保留最后一个 chunk 的 `usage_metadata`，并在最终事件回传。

### 6. Observability

之前 observability 只稳定记录：

- `inputTokens`
- `outputTokens`
- `totalTokens`

现在 summary 和 llm 事件还会记录：

- `cachedInputTokens`
- `reasoningTokens`
- `cacheReadTokens`
- `cacheCreationTokens`
- `toolUsePromptTokens`

并且 `inputTokens` 在 provider 返回真实值时会覆盖掉原来的请求前估算值，不再出现“input 是估算、output 是真实”的混合口径。

## 一个具体过程例子

以 `openai chat` 为例，现在线路是：

1. `llm = EasyLLM(provider="openai", ...)`
2. `response = llm.invoke_raw([...])`
3. `usage = llm.extract_usage_metrics(response)`
4. `agent.stream_invoke(...)`
5. `agent.get_recent_observability_events(event_type="llm")`

以前第 3 步经常只能拿到空 dict，第 5 步里的 token 也大概率是估算值。

现在：

- 第 3 步直接拿真实 provider usage
- 第 5 步里的 `inputTokens/outputTokens/totalTokens/cachedInputTokens/reasoningTokens`
  会跟 provider 返回值保持一致

## 这次最重要的结果

EasyAgent 现在的 usage 链路从“统一猜测器”变成了“provider-first，fallback-second”：

- 真实 provider usage 优先
- 只在拿不到真实 usage 时才估算
- stream 和 non-stream 都打通了
- `BasicAgent` 这一层不会再把 stream usage 丢掉

## 对上层的意义

这次改完后，上层 code agent / benchmark / cost accounting 可以直接建立在框架返回的真实 usage 上，而不是自己再对 response schema 做一层猜测适配。

尤其是这几类场景会直接受益：

- benchmark 对比真实 `input/output/total`
- prompt caching 效果统计
- reasoning token 成本分析
- stream 场景下的真实 token 记账
