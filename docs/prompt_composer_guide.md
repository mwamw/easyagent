# System Prompt Composer Guide

`SystemPromptComposer` 是系统提示词的默认实现。最简单的定制只需要创建 `PromptBlock`；动态定制时，继承 `BaseSystemPromptComposer` 并实现 `build(context)`。

## 1. PromptBlock

```python
from easyagent import PromptBlock

PromptBlock(
    name="product_policy",
    content="先读取事实，再给出结论。",
    placement="system",
    order=100,
)
```

字段：

- `name`：稳定名称，同名自定义块覆盖默认块。
- `content`：字符串，或接收 `PromptBuildContext` 的函数。
- `placement`：`system` 或 `system_reminder`。
- `order`：渲染顺序。
- `enabled`：是否启用。
- `metadata`：缓存分区等附加信息。

`system` 进入 provider 原生 system。`system_reminder` 包装成 `<system-reminder>`，只加入当前 request 的 prepended replay，不写入 canonical history。

## 2. 直接配置

```python
from easyagent import PromptBlock, SystemPromptComposer

composer = SystemPromptComposer(
    blocks=[
        PromptBlock(
            "identity",
            "你是一个游戏开发 Agent。",
            placement="system",
        ),
        PromptBlock(
            "workspace",
            lambda ctx: f"当前工作区：{ctx.execution_context.workspace_root}",
            placement="system_reminder",
            order=100,
        ),
    ],
    include_defaults=True,
)
agent.with_prompt(composer)
```

`include_defaults=True` 时，框架保留默认安全、工具和输出规则；同名块可以覆盖其中某一项。设为 `False` 时完全由用户负责系统提示词。

## 3. 自定义 Composer

```python
from easyagent import BaseSystemPromptComposer, PromptBlock, PromptBuildContext


class ProductComposer(BaseSystemPromptComposer):
    def build(self, context: PromptBuildContext) -> list[PromptBlock]:
        return [
            PromptBlock(
                "identity",
                f"你是 {context.agent_name}，负责软件工程任务。",
            ),
            PromptBlock(
                "mode",
                f"当前模式：{context.execution_context.execution_mode}",
                placement="system_reminder",
            ),
        ]


agent.with_prompt(ProductComposer())
```

用户不需要实现 provider 消息格式、工具 inventory、request compiler 或 history 接线。

## 4. PromptBuildContext

可读取：

- `agent_name`
- `description`
- `system_prompt`
- `query`
- `config`
- `execution_context`
- `tool_registry`
- `skill_manager`
- `memory`
- `task_service`
- `plan`

Context 不包含 Agent 反向引用。动态 content 应只读取状态，不执行网络、文件写入或 mailbox 消费。

## 5. 默认 Composer

默认实现根据已安装模块生成：

- identity、visibility、task execution、安全、语气和输出规则
- Tool 已安装时的工具策略
- Skill policy、listing 和 resident skill body
- Memory 已安装时的稳定使用规则
- deferred tool inventory

Skill listing 和 deferred inventory 使用 `system_reminder`；稳定行为规则使用 `system`。

## 6. 调试

```python
template = agent.get_system_prompt_template("当前 query")
native_system = template.render_system()
request_reminders = template.render_system_reminders()
all_blocks = template.get_blocks()
```

`agent.get_enhanced_prompt(query)` 只返回 native system。

## 7. 与 MetaMessage 的边界

PromptBlock 用于每次请求重新组装的系统级指令。MetaMessage 用于写入 history 的运行时事件或临时条件指令。

- 当前工作区摘要：system reminder。
- 进入/退出 plan：MetaMessage。
- mailbox 到达：MetaMessage。
- invocation 期间启用的 Skill body：MetaMessage。
- 工具临时结果：MetaMessage。

不要把 MetaMessage 当成第三种 placement；它们是不同模块。
