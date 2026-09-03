# Prompt And MetaMessage Guide

EasyAgent 把“稳定的系统规则”和“运行时事件产生的指令”分成两个平级模块：

- `SystemPromptComposer` 组装每次 request 的 system prompt。
- `MetaMessageManager` 把动态指令作为 canonical user message 追加到 history 末尾，并按 lifecycle 回收。

两者不互相兼容、不共享 placement，也不会将 Agent 实例传给用户的组装函数。

## 1. SystemPromptComposer

`PromptBlock` 有两种 placement：

| placement | 发送位置 | 是否进入 canonical history |
| --- | --- | --- |
| `system` | provider 原生 system | 否 |
| `system_reminder` | 当前 request 的 prepended user reminder | 否 |

```python
from easyagent import BasicAgent, PromptBlock, SystemPromptComposer

composer = SystemPromptComposer(
    blocks=[
        PromptBlock(
            name="identity",
            content="你是一个谨慎的代码审查 Agent。",
            placement="system",
            order=0,
        ),
        PromptBlock(
            name="workspace",
            content=lambda ctx: f"当前工作区：{ctx.execution_context.workspace_root}",
            placement="system_reminder",
            order=100,
        ),
    ],
    include_defaults=True,
)

agent = BasicAgent(name="reviewer", llm=llm).with_prompt(composer)
```

`content` 函数接收 `PromptBuildContext`，其中只有组装提示词需要的请求级数据，没有 Agent 反向引用。

### 自定义 Composer

复杂产品可继承 `BaseSystemPromptComposer`，只需实现 `build(context)`：

```python
from easyagent import BaseSystemPromptComposer, PromptBlock


class GameAgentPrompt(BaseSystemPromptComposer):
    def build(self, context):
        return [
            PromptBlock("identity", "你是游戏引擎开发 Agent。"),
            PromptBlock(
                "mode",
                f"当前模式：{context.execution_context.execution_mode}",
                placement="system_reminder",
            ),
        ]
```

## 2. MetaMessage

MetaMessage 不属于 system prompt placement，也不是通过 `with_*` 安装的用户模块。它是 Agent 内建的模块开发 SPI：功能模块在真实事件触发后调用 `agent.emit_metamessage(...)`，Manager 再生成 `CanonicalMessage(role="user")` 并插入对话末尾。

普通 Agent 使用者应通过 SystemPromptComposer 或 `with_skill()`、`with_plan()` 等具体模块表达行为，不直接管理 MetaMessage。自定义模块负责判断触发条件，并在触发时构造相应生命周期的 MetaMessage。

Lifecycle：

- `PERMANENT`：注入后保留在 history，用于 mailbox 或模式转换记录。
- `INVOCATION`：在一次 `invoke/stream` 期间有效，结束时自动回收。
- `REQUEST`：只在一次 LLM request 期间有效，该次 LLM 调用成功或失败后都会回收。
- `CONDITIONAL`：条件从 false 变为 true 时注入，回到 false 时自动移除。

## 3. 框架模块如何使用 MetaMessage

- Plan 模式进入时注入一条永久 `plan_mode_enter`，退出时再注入 `plan_mode_exit`。
- 按需 Skill 被工具加载时注入 `INVOCATION` 消息，本次 Agent invoke 结束后回收。
- MultiAgent mailbox 在子 Agent 每次 request 前同步，每条消息通过 dedup key 只注入一次。
- 工具返回的 `ephemeral_context` 作为 request/invocation 上下文管理，不需要扩展 SystemPromptComposer。

## 4. 请求组装顺序

1. Executor 开始 invocation，MetaMessageManager 记录当前 query 和运行状态。
2. MultiAgent 同步 mailbox，Skill/Plan/Tool 可发布新 MetaMessage。
3. Manager 在 request 安全点 `flush()`，将消息写入 canonical history。
4. SystemPromptComposer 生成 system 与 system reminder blocks。
5. Context pipeline 添加检索结果等 dynamic context。
6. Provider codec 把 canonical history 转换为目标 provider 格式。
7. request/invocation 结束时，Manager 回收对应 lifecycle 的消息。

## 5. 边界选择

- 长期、稳定、与运行时无关的规则放 SystemPrompt。
- 需要 provider 原生 system 语义的内容使用 `placement="system"`。
- 每次 request 都需重新计算、但不应进入 history 的提示使用 `system_reminder`。
- 由运行时事件触发、需要影响后续对话的内容使用 MetaMessage。
- 不要在 Composer 中直接修改 Agent，也不要在 MetaMessage 中模拟 system placement。

更详细的自定义接口见 [System Prompt Composer Guide](./prompt_composer_guide.md) 和 [Modular Agent Architecture](./modular_agent_architecture.md)。
