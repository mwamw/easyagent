# Prompt System Guide

EasyAgent 的 prompt 不是“一个拼好的字符串”，而是一套分层请求结构。  
理解 prompt 系统，等于理解：

- system prompt 该放什么
- runtime reminder 该放什么
- skill body / tool schema 该放什么
- 为什么 cache 命中率会变化

相关文档：

- [Prompt Composer Guide](./prompt_composer_guide.md)
- [Runtime Reminders Guide](./runtime_reminders_guide.md)
- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 1. 四层结构

当前主路径把请求分成四层：

### `system_core`

放稳定内容：

- identity
- 长期行为规则
- 安全策略
- tool policy
- 输出风格

### `runtime_reminders`

放每次请求 prepend 一次、但不写入长期 history 的运行时上下文：

- 产品壳层说明
- 当前环境
- 能力概览
- 稳定的 tool / skill listing

### `on_demand_expansion`

放按需展开内容：

- deferred tool schemas
- 临时 skill body

### `dynamic_tail`

放动态内容：

- memory retrieval
- mailbox
- 当前轮 runtime skill context
- 当前 query 相关附加上下文

## 2. 不应该把什么塞进 system_core

通常不应该直接放进 `system_core`：

- 当前日期
- 动态 memory
- mailbox
- 当前轮临时 skill 正文
- 经常变化的 UI 壳层状态

这些更适合：

- runtime reminders
- dynamic tail

## 3. 最简单的系统提示词定制方式

### 方式一：直接传 `system_prompt`

```python
agent = BasicAgent(
    name="assistant",
    llm=llm,
    system_prompt="你是一个谨慎的工程助手。",
)
```

### 方式二：注入 `PromptBlock`

```python
from easyagent.prompting import PromptBlock

agent.with_prompt_block(
    PromptBlock(
        name="product_policy",
        content="优先中文回答；先给结论，再给理由。",
        order=120,
    )
)
```

适合：

- 追加产品规则
- 插件扩展
- 不想重写整个 composer 的场景

### 方式三：自定义 Prompt Composer

详见：

- [Prompt Composer Guide](./prompt_composer_guide.md)

## 4. Tool 与 Skill 如何影响 prompt

### Tool

- tool inventory 可以进入 reminder 层
- deferred tool schema 进入 on-demand expansion

### Skill

- skill listing 可以进入 reminder 或 shared prompt
- turn skill body 通常进入 dynamic tail 或 on-demand expansion

## 5. 和 Agent 的集成方式

你通常会通过这些入口修改 prompt：

- `system_prompt=...`
- `agent.with_prompt_block(...)`
- `agent.with_prompt_blocks(...)`
- 自定义 `prompt_composer`
- `agent.with_runtime_reminder(...)`
- `agent.add_runtime_reminder_source(...)`

## 6. 与 cache 的关系

Prompt 结构直接决定 cache 能不能稳定命中。

经验原则：

- 稳定规则放前面
- 动态内容后移
- 重型 schema 按需展开

如果系统提示词越来越长，不代表一定更好。  
比“写更多规则”更重要的是“把规则放在正确层里”。
