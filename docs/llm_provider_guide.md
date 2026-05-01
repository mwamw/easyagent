# LLM Provider Guide

`EasyLLM` 是 EasyAgent 的统一模型访问层。  
它把不同 provider 的请求、tool-calling、streaming、usage、thinking 与 request buffer 组织成统一接口。

相关文档：

- [Agent Guide](./agent_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 1. 最小初始化

```python
from easyagent import EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="x",
    model="qwen3.5-9b",
)
```

## 2. 主要参数

- `model`
  - 模型名
- `temperature`
  - 温度
- `max_tokens`
  - 最大输出 token
- `api_key`
  - API key
- `base_url`
  - 接口地址
- `timeout`
  - 请求超时
- `provider`
  - provider 名称
- `**kwargs`
  - provider-specific 参数

## 3. 常见 provider

### `openai`

适用于 OpenAI Chat Completions 兼容接口或兼容网关。

特点：

- 生态广
- usage 常见字段是 `prompt_tokens / completion_tokens / cached_tokens`
- 更依赖结构稳定化而不是显式 cache marker

### `openai_responses`

适用于 Responses 风格接口。

特点：

- 输出结构和 chat 接口不同
- usage 字段也可能不同

### `anthropic_native`

适用于原生 Anthropic / Claude 风格接口。

特点：

- message-level / system-level cache marker 能力最明确
- thinking / tool-calling / cache 语义最完整

### `google_native`

适用于原生 Google Gemini 风格接口。

特点：

- 有 `thought_signature` 这类 provider-specific 结构
- cached content 能力要看后端是否真实支持

## 4. 自动识别 vs 显式指定

`provider="auto"` 时，`EasyLLM` 会根据模型名和环境变量尝试识别 provider。  
产品中更推荐显式指定 provider，减少网关兼容层差异带来的歧义。

## 5. 和 Agent 的集成

```python
agent = BasicAgent(name="assistant", llm=llm)
```

在 tool agent 中：

```python
agent = BasicAgent(name="tool-agent", llm=llm, enable_tool=True, tool_registry=registry)
```

## 6. tool-calling 与 request buffer

`EasyLLM` 不只是发字符串。  
它会消费 `ReplayRequestInput`，并根据 provider：

- 组装 system
- 组装 messages / contents
- 注入 tools schema
- 记录 usage
- 在支持时接 cache adapter

## 7. usage 与 cache 语义

不同 provider 返回的 usage 口径不同：

- Anthropic 风格：`inputTokens` 与 `cacheReadTokens`
- OpenAI 风格：`prompt_tokens` 与 `cached_tokens`
- Google 风格：可能返回 prompt / candidate / thought token，但不一定带 cache 字段

EasyAgent 会把它们归一化到 observability 层。详见：

- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 8. 调试建议

遇到 provider 问题时优先确认：

1. `provider`
2. `model`
3. `base_url`
4. `api_key`
5. 请求格式是否和该 provider 匹配
6. usage / raw response 是否被正确记录

如果是 Google Native：

- 不要随意改写带 `thought_signature` 的历史结构

如果是 Anthropic：

- 检查 `cache_control` 是否真的落在预期层

如果是 OpenAI / NewAPI：

- 区分“字段存在”与“真实命中”
