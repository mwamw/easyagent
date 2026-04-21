# EasyAgent 各 Provider 详细对比文档

本文档为开发者提供 EasyAgent 框架支持的不同 LLM Provider（服务商）之间的深度技术对比。涵盖了底层的 API 协议、多轮对话的消息负载、推理/思考过程（Thinking/Reasoning）的表征形式、工具调用的生命周期以及流式输出的增量细节。

## Provider 系统架构

EasyAgent 通过 `Provider` 指令集和 `Codec` 解析层对不同服务商进行了抽象，目标是在保持原生特性的同时，提供统一的 `CanonicalMessage` 表达：

1.  **OpenAI 兼容型 (`openai`, `openai_compat`)**: 适配标准的 Chat Completions API。包括官方 OpenAI、DeepSeek、Qwen、Kimi、Ollama 等几乎所有兼容型服务。
2.  **OpenAI Responses (`openai_responses`)**: 专为 OpenAI 的新一代模型（如 o1, o3 系列）设计的 "Typed Items" 协议。
3.  **Google Native (`google_native`)**: 直接调用 Google Gemini API，支持原生的多模态和思考块处理。
4.  **Anthropic Native (`anthropic_native`)**: 直接调用 Anthropic Claude Messages API，深度适配 Claude 3.5/3.7 系列的思考能力。

---

## 核心参数对比

| 维度 | OpenAI 兼容型 | OpenAI Responses | Google Native | Anthropic Native |
| :--- | :--- | :--- | :--- | :--- |
| **主要入参字段** | `messages` | `input` | `contents` | `messages` |
| **系统指令 (System)** | `role: system` 消息 | `instructions` 字段 | `system_instruction` 字段 | `system` 字段 |
| **推理/思维配置** | `reasoning_effort` | `reasoning` 配置对象 | `thinking_config` | `thinking` 配置对象 |
| **模型回复角色** | `assistant` | `assistant` (Typed Item) | `model` | `assistant` |
| **最大 Token** | `max_tokens` | N/A (模型自动决策) | `max_output_tokens` | `max_tokens` |

---

## 推理与思考过程 (Thinking & Reasoning)

这是各家差异最大的地方。EasyAgent 将这些内容统一映射为 `CanonicalBlock(type="reasoning")`。

### 1. Anthropic (Claude 3.7+)
Claude 的思考内容作为回复原文的一部分，以 `thinking` 块形式存在，并且带有系统签名。
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "我需要先分析用户的代码结构...",
      "signature": "sk_anthropic_abc123..."
    },
    {
      "type": "text",
      "text": "分析完毕，代码如下："
    }
  ]
}
```

### 2. Google (Gemini 2.0+)
Gemini 的 `parts` 列表中，思考块通过 `thought: true` 标记。
```json
{
  "role": "model",
  "parts": [
    {
      "text": "正在检索相关文件以确定接口定义...",
      "thought": true,
      "thought_signature": "gemini_sig_xyz..."
    },
    {
      "text": "找到定义，接口名为 `run_task`。"
    }
  ]
}
```

### 3. OpenAI Responses (o1, o3)
OpenAI o1 使用独立的层级结构，思考过程可能包含加密数据（Redacted）。
```json
{
  "output": [
    {
      "type": "reasoning",
      "summary": [{"type": "summary_text", "text": "分析中..."}],
      "signature": "o1_sig_789...",
      "content": "具体的思考逻辑内容..."
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "你好！"}]
    }
  ]
}
```

---

## 工具调用生命周期 (Tool Calls & Lifecycle)

### 1. 调用请求 (Assistant Turn)
| Provider | 结构特征 |
| :--- | :--- |
| **OpenAI** | 扁平的 `tool_calls` 列表。 |
| **Google** | `parts` 列表中的 `function_call` 对象，包含 `id`, `name`, `args`。 |
| **Anthropic** | `content` 列表中的 `tool_use` 块，包含 `id`, `name`, `input`。 |
| **Responses** | 独立的 `type: "function_call"` 条目。 |

### 2. 结果回传 (User/Tool Turn)
这是极易出错的部分，必须精准遵循各家协议：

#### A. Anthropic (Claude) - 强约束：必须包装在 `user` 角色中
```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01...",
      "content": "执行成功：已创建目录。",
      "is_error": false
    }
  ]
}
```

#### B. Google (Gemini) - 强约束：角色为 `user`，内容为 `parts`
```json
{
  "role": "user",
  "parts": [
    {
      "function_response": {
        "id": "call_abc",
        "name": "create_dir",
        "response": {"result": "success", "path": "/home/user"}
      }
    }
  ]
}
```

#### C. OpenAI (Standard) - `tool` 角色
```json
{
  "role": "tool",
  "tool_call_id": "call_abc",
  "name": "create_dir",
  "content": "success"
}
```

#### D. OpenAI Responses - 独立的 `function_call_output` 条目
```json
{
  "type": "function_call_output",
  "call_id": "call_abc",
  "output": "success"
}
```

---

## 流式响应 (Streaming) 细节

流式输出时，各家发送的增量块（Delta）结构迥异，EasyAgent 进行了平滑处理。

| 事件类型 | OpenAI (Delta) | Anthropic (Event) | Google (Chunk) |
| :--- | :--- | :--- | :--- |
| **文本增量** | `delta.content` | `content_block_delta` (text) | `candidates[0].content.parts[0].text` |
| **思考增量** | `delta.reasoning_content` | `content_block_delta` (thinking) | 带有 `thought: true` 的 Part 增量 |
| **思考签名** | 消息末尾下发 | `signature_delta` 事件 | `thought_signature` 字段 |
| **工具增量** | `delta.tool_calls` (分段 index) | `input_json_delta` 事件 | N/A (通常完整下发 Part) |

---

## 最佳实践与注意事项

1.  **角色映射**：EasyAgent 会自动将 Google 的 `model` 映射为 `assistant`。在手写提示词或拦截消息时，注意这种角色的双向转换。
2.  **思绪丢失风险**：部分兼容型 Provider（如通过代理转接的 Claude）可能会丢失 `thinking` 块。EasyAgent 推荐使用原生 SDK Provider 以获得最佳体验。
3.  **多块共存**：Claude 和 Gemini 支持在一次回复中同时发送文本和工具调用。EasyAgent 会将其解析为包含多个 `CanonicalBlock` 的单一 `CanonicalMessage`。
4.  **Token 统计**：推理 Token (Reasoning Tokens) 在不同计费系统中有显著区别，请通过 `agent.get_context_usage()` 获取精确的分解统计。
