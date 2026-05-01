# Runtime Reminders Guide

Runtime Reminder 是 EasyAgent 用来承载“运行时高可信上下文”的机制。  
它的定位类似“每次请求前 prepend 一次的结构化提醒”，而不是长期写进 canonical history 的普通消息。

相关文档：

- [Prompt System Guide](./prompt_system_guide.md)
- [Prompt Composer Guide](./prompt_composer_guide.md)

## 1. 它解决什么问题

很多产品上下文本来不该写进 system core，例如：

- 当前产品 shell 说明
- slash command 约定
- 当前环境
- 当前稳定能力列表

如果把这些东西硬塞进 system：

- system prompt 会频繁抖动
- cache 更难命中

Runtime Reminder 的目标就是把这类内容单独放到 request-time reminder 层。

## 2. 核心对象

### `RuntimeReminder`

单个 reminder。

关键字段：

- `name`
- `content`
- `order`
- `stable`
- `cacheable`
- `metadata`

### `BaseRuntimeReminderSource`

可扩展的 reminder 生成器。  
适合做：

- 产品外壳上下文
- 固定 UI 环境注入
- 多来源 reminder 聚合

### `StaticRuntimeReminderSource`

最简单的固定文本 reminder source。

## 3. 最小示例

```python
agent.with_runtime_reminder(
    name="product_shell",
    content="你运行在一个带 slash command 的 IDE 面板中。",
)
```

或：

```python
from easyagent.reminders import BaseRuntimeReminderSource, RuntimeReminder

class ProductContextSource(BaseRuntimeReminderSource):
    def build_runtime_reminders(self, agent):
        return [
            RuntimeReminder(
                name="product_shell",
                content="你运行在一个带 slash command 的 IDE 面板中。",
            )
        ]

agent.add_runtime_reminder_source(ProductContextSource())
```

## 4. stable vs dynamic

### `stable=True`

适合稳定提醒。  
例如：

- 产品定位
- 工具目录提示
- 基本交互规则

### `stable=False`

适合更动态的提醒。  
例如：

- 短期 UI 状态
- 当前一次性上下文

## 5. 它和 history 的关系

Reminder 会在每次请求里 prepend 一次，但不写进长期 canonical history。  
这意味着：

- 能稳定影响本次请求
- 不会在历史里重复堆叠

## 6. 和 Agent 的集成方式

### 单条 reminder

```python
agent.with_runtime_reminder(name="x", content="...")
```

### 多来源 reminder

```python
agent.add_runtime_reminder_source(source)
```

### 结合 Prompt Composer

composer 可以产出 reminder block，但不应独占运行时上下文的全部逻辑。  
职责上，source 更适合承载产品态上下文。
