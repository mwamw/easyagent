# Tool System Guide

Tool 系统是 EasyAgent 的执行层。  
它由 `Tool`、`ToolRegistry`、权限判定、确认中断、工具循环以及大量内置工具构成。

相关文档：

- [Builtin Tools Catalog](./builtin_tools_catalog.md)
- [Tool Authoring Guide](./tool_authoring_guide.md)
- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Permissions Guide](./permissions_guide.md)

## 1. 核心对象

### `Tool`

单个工具的定义，负责：

- 参数 schema
- 读写 / 副作用语义
- 运行逻辑
- deferred 默认暴露语义

### `ToolRegistry`

工具注册表，负责：

- 注册 / 覆盖 / 删除工具
- 维护 visibility
- 导出 provider tools payload
- deferred expansion
- 参数校验
- 授权执行

### `ToolResult`

标准化工具结果。  
推荐用：

- `ToolResult.success(...)`
- `ToolResult.error(...)`
- `ToolResult.needs_confirmation(...)`

## 2. 最小自定义示例

```python
from pydantic import BaseModel
from easyagent.tools import Tool, ToolResult, ToolRegistry

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
        )

    def run(self, parameters: dict):
        value = parameters["a"] + parameters["b"]
        return ToolResult.success(str(value), structured_data={"value": value})

registry = ToolRegistry()
registry.register_tool(AddTool())
```

## 3. ToolRegistry 负责什么

`ToolRegistry` 的重点职责：

- 注册工具
- 维护 `resident / runtime / turn` 三类 visibility
- 稳定排序导出工具
- `full / deferred` 两种 schema 模式
- 执行前参数校验
- 结合 `PermissionEngine` 生成 Allow / Ask / Deny
- 标准化工具结果

## 4. 工具可见性

### `resident`

常驻工具。  
多数 builtin tools 和产品常用工具属于这一类。

### `runtime`

当前运行期可见。  
常见于：

- skill 临时挂载工具
- runtime manager 注入工具

### `turn`

当前轮可见。  
适合短生命周期工具。

## 5. 工具执行路径

一个 tool call 的典型路径是：

1. 模型返回 tool call
2. `ToolRegistry.validate_tool_call(...)`
3. `ToolRegistry.authorize_tool_call(...)`
4. 若需确认，返回 `ToolResult.needs_confirmation(...)`
5. 若允许，执行 `tool.run(...)`
6. `ToolRegistry.normalize_tool_result(...)`
7. 结果写回当前 invoke

## 6. deferred vs full

### `full`

初始请求直接暴露所有完整 schema。

### `deferred`

初始请求不全量暴露所有 schema，而是：

- 只暴露默认暴露工具
- 使用 `tool_schema_tool` 按需展开其他工具

详见：

- [Deferred Tools Guide](./deferred_tools_guide.md)

## 7. 如何把 tool 接到 agent

```python
agent = BasicAgent(
    name="tool-agent",
    llm=llm,
    enable_tool=True,
    tool_registry=registry,
)
```

常见产品组合：

- `register_filesystem_tools(...)`
- `register_shell_tools(...)`
- `register_task_tools(...)`
- `register_agent_tool(...)`
- `register_mailbox_tools(...)`

## 8. 风险与确认

Tool 的风险控制由两层组成：

1. Tool 自身的语义字段
   - `read_only`
   - `destructive`
   - `requires_confirmation`
   - `risk_categories`
2. PermissionEngine 根据上下文做最终决策

如果一个工具需要用户确认，Agent 通常会中断当前 invoke，把确认请求交给上层应用处理。

## 9. 什么时候该写自定义工具

适合写自定义工具的情况：

- 你有业务系统 API
- 你要做产品级 DSL 或动作封装
- 你不想让模型直接面对底层 shell / 文件系统

不适合写成工具的情况：

- 只是想追加一点提示词
- 只是想加运行时上下文
- 更适合写成 skill 或 reminder
