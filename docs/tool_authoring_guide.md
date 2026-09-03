# Tool Authoring Guide

这份文档专门解释如何自定义 Tool，以及 `Tool(...)` 构造函数、执行路径、返回值、deferred 暴露和权限相关字段的真实意义。  
如果你要基于 EasyAgent 做产品，这份文档应该回答两个问题：

- 我什么时候应该写一个 Tool？
- 我写 Tool 时每个构造参数到底在影响什么？

相关文档：

- [Tool System Guide](./tool_system_guide.md)
- [Builtin Tools Catalog](./builtin_tools_catalog.md)
- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Permissions Guide](./permissions_guide.md)

## 1. 什么场景应该写成 Tool

优先写成 Tool 的场景：

- 模型需要主动调用某项能力
- 这项能力有结构化参数
- 这项能力需要权限、确认、风险标记
- 你希望这项能力在日志、回调、观测里被单独记录

不要写成 Tool 的场景：

- 只是系统规则
- 只是当前请求的上下文
- 只是一个不需要模型主动调用的内部辅助函数

可以这样理解：

- Tool 解决“调用能力”
- Prompt / Reminder 解决“告诉模型规则或上下文”
- Skill 解决“组织一组相关能力”

## 2. 最小工具示例

```python
from pydantic import BaseModel
from Tool.BaseTool import Tool, ToolResult


class AddParams(BaseModel):
    a: int
    b: int


class AddTool(Tool):
    def __init__(self):
        super().__init__(
            name="add",
            description="计算两个整数之和",
            parameters=AddParams,
            read_only=True,
            expose_in_deferred=True,
        )

    def run(self, parameters: dict):
        value = parameters["a"] + parameters["b"]
        return ToolResult.success(
            str(value),
            structured_data={"value": value},
        )
```

这个例子已经包含了一个 Tool 的最小闭环：

- Pydantic 参数模型
- Tool 实例
- 同步 `run`
- `ToolResult.success(...)`

## 3. `Tool(...)` 构造参数分组解释

下面按功能分组说明。不是每个 Tool 都需要全部字段，但你至少应该知道它们在框架里分别影响什么。

### 3.1 标识与 schema

#### `name`

- 工具唯一名字。
- 模型看到的 tool name、日志里的 tool name、registry 里的 key 都依赖它。
- 应尽量稳定，不要轻易改名。

#### `description`

- 给模型和开发者看的工具描述。
- 会进入 tool schema 或 inventory/listing。
- 要写“做什么”和“适合什么时候用”，不要只写技术实现。

#### `parameters`

- Pydantic 参数模型。
- 决定：
  - 模型能填哪些字段
  - 参数如何校验
  - JSON schema 如何导出

#### `guidance`

- 附加的使用指导。
- 适合放：
  - 调用前约束
  - 参数选择建议
  - 结果解释方式

不要把核心产品逻辑全塞进 `guidance`，否则 schema 会变重。

### 3.2 风险与执行行为

#### `read_only`

- 标记工具是否只读。
- 影响：
  - 风险分级
  - 权限系统默认判断
  - 某些产品 UI 的展示方式

#### `destructive`

- 标记工具是否具有明显破坏性。
- 适合：
  - 删除
  - 回滚
  - 覆盖
  - 停止外部进程

#### `requires_confirmation`

- 表示调用前需要用户确认。
- 这不是 UI 本身，而是框架层向上层应用暴露“这里应该 ask”的信号。

#### `supports_parallel`

- 表示这个工具是否允许并行调用。
- 对远程资源竞争或全局共享状态工具很重要。

#### `output_mode`

- 描述工具输出应该怎样被消费。
- 主要给框架或产品侧做渲染/协议上的提示。

### 3.3 生命周期与来源

#### `source`

- 工具来源标记，例如：
  - `builtin`
  - `custom`
  - `mcp`
- 它是来源信息，不应该承担核心业务语义。

#### `ephemeral`

- 表示这是临时工具。
- 常见于：
  - runtime skill 挂载工具
  - 当前 invoke 临时展开的能力

#### `visibility_scope`

- 当前工具属于：
  - `resident`
  - `runtime`
  - `turn`

推荐理解：

- `resident`
  - 长期存在
- `runtime`
  - 当前 runtime 生命周期内存在
- `turn`
  - 当前 invoke 或当前轮内存在

### 3.4 Prompt / Skill 关联字段

#### `prompt`

- 工具附带的补充说明文本。
- 适合工具自身的特殊注意事项。

#### `prompt_visibility`

- 较早期字段。现在主路径不建议依赖它来设计关键逻辑。

#### `expose_in_deferred`

- 表示 deferred schema 模式下是否默认暴露该工具。
- Skill 的 `allowed-tools` 可以在当前 invoke 中临时展开未默认暴露的工具。

### 3.5 风险标签与产品元信息

#### `tags`

- 纯标签，适合分类和调试。

#### `risk_categories`

- 风险类别标签，供 permission / policy / UI 使用。

#### `side_effect_level`

- 副作用等级提示。

#### `resource_scope`

- 描述它影响的是哪类资源。

这些字段通常不会单独决定执行，但会影响：

- 权限判定
- 审计日志
- UI 呈现

### 3.6 Deferred 相关

#### `expose_in_deferred`

- 在 `tool_schema_mode="deferred"` 下是否默认暴露。
- 这是 deferred 默认暴露行为的正式字段。

推荐用法：

- 高频基础工具：`True`
- 低频重型工具：`False`
- 高风险或需要先缩小范围的工具：通常 `False`

不要再用：

- `metadata["always_exposed"]`
- 其他隐式约定

来承载这类核心语义。

### 3.7 `metadata`

- 留给产品自定义的扩展字段。
- 可以放：
  - 业务标签
  - UI 呈现建议
  - 额外审计字段

不建议把框架核心逻辑写进 `metadata`，否则会变得难以维护。

## 4. `run` 和 `arun` 怎么选

### `run(parameters: dict)`

- 同步执行入口。
- 最少必须实现它之一。

适合：

- 本地快速操作
- 轻量读写
- 同步 API

### `arun(parameters: dict)`

- 异步执行入口。

适合：

- 远程异步调用
- 网络操作
- 需要并发等待的场景

一般建议：

- 你的工具天然异步，就实现 `arun`
- 否则先用 `run`

## 5. `ToolResult` 怎么用

推荐统一返回 `ToolResult`，而不是裸字符串。

最常见三种：

```python
ToolResult.success(...)
ToolResult.error(...)
ToolResult.needs_confirmation(...)
```

这样做的好处是：

- permission engine 更容易接
- callback/observability 能拿到结构化结果
- UI 更好渲染
- runtime 更容易判断后续动作

### `success`

适合正常完成。

### `error`

适合工具执行失败，但错误是可解释的业务错误。

### `needs_confirmation`

适合工具本身判断：

- 这是高风险动作
- 需要先 ask，再 allow

## 6. 一次典型执行流程

一个 Tool 真正被调用时，通常会经过这些层：

1. 模型选择某个工具
2. `ToolRegistry` 找到对应 Tool
3. 参数经过 schema 校验
4. permission / hook / guardrail 做前置判定
5. 调用 `run` 或 `arun`
6. 结果包装成 `ToolResult`
7. callback / observability 记录事件
8. 结果回放给模型

这也是为什么 Tool 不只是一个 Python 函数：它还承担协议、权限和运行时边界。

## 7. Deferred 模式下如何设计工具

Deferred 模式的关键不是“所有工具永远不暴露”，而是：

- 初始不全量暴露 schema
- 只暴露必要的工具目录和关键基础工具
- 真需要时再展开 schema

设计建议：

### 默认应暴露

- 高频基础工具
- 工具目录/调度工具
- 能帮助模型继续发现能力的工具

### 默认不应暴露

- 低频重型工具
- 高副作用工具
- 很少会在第一跳就需要的工具

## 8. 如何注册到 `ToolRegistry`

最常见方式：

```python
from Tool.ToolRegistry import ToolRegistry

registry = ToolRegistry()
registry.register_tool(MyTool())
```

如果要覆盖 deferred 行为：

```python
registry.register_tool(MyTool(), expose_in_deferred=False)
```

## 9. 如何接到 `BasicAgent`

```python
from easyagent import BasicAgent, EasyLLM
from Tool.ToolRegistry import ToolRegistry

registry = ToolRegistry()
registry.register_tool(MyTool())

agent = BasicAgent(
    name="assistant",
    llm=EasyLLM(),
).with_tool(registry)
```

## 10. 推荐实践

### 工具描述写“作用 + 适用时机”

不要只写技术名词。

### 参数模型尽量清晰

用 Pydantic 明确字段名、类型和描述，不要让模型猜。

### 高风险动作显式标记

不要把确认逻辑全扔给上层 UI 猜。

### Tool 尽量单一职责

一个 Tool 只做一类动作，便于模型理解和权限审计。

### 临时工具显式标成 runtime/turn

否则 invoke 结束后不容易清理。

## 11. 常见坑

### 坑一：用 `metadata` 承载框架核心逻辑

例如用它决定 deferred 默认暴露，这会让语义非常隐晦。

### 坑二：把一整套流程都塞进一个 Tool

模型更难稳定调用，权限边界也更模糊。

### 坑三：返回裸字符串

这样会失去很多结构化运行时能力。

### 坑四：只考虑本地执行，不考虑权限和观测

产品化时很快会补不回来。

### 坑五：把所有工具都设成 `expose_in_deferred=True`

这样等于把 deferred 模式重新退化成全量 schema 暴露。
