# EasyAgent Tool 工具系统使用指南

> **版本**: 1.1.0 | **最后更新**: 2026-04-12

## 目录

- [概述](#概述)
- [当前 Tool 模型](#当前-tool-模型)
- [核心架构](#核心架构)
- [快速开始](#快速开始)
- [Tool 基类详解](#tool-基类详解)
- [ToolSpec 说明](#toolspec-说明)
- [ToolResult 说明](#toolresult-说明)
- [ToolRegistry 管理器](#toolregistry-管理器)
- [与 Agent / Skill 的集成](#与-agent--skill-的集成)
- [内置工具](#内置工具)
- [设计约束与最佳实践](#设计约束与最佳实践)
- [API 参考](#api-参考)

---

## 概述

Tool（工具）系统是 EasyAgent 的统一能力调用层。它负责把“模型可调用的函数能力”封装成标准对象，并接到：

- LLM 的 function calling / tool calling
- Agent 的工具执行循环
- Skill 的 resident / on-demand 能力挂载
- Memory / MCP / Search 等内置能力

每个 Tool 由三部分组成：

| 组成 | 说明 |
|------|------|
| `ToolSpec` | 工具元数据，描述名称、参数模型、风险属性、来源等 |
| `run(parameters)` | 工具的实际执行逻辑 |
| `ToolResult` | 工具执行结果协议，统一成功/失败/确认/结构化数据返回 |

---

## 当前 Tool 模型

从 1.1 开始，EasyAgent 的 Tool 系统有三个关键变化：

### 1. Tool prompt 不再进入 system prompt

当前设计已经和 Claude Code 的主思路对齐：

- 工具的详细说明通过 `tools` 参数里的 schema `description` 传给模型
- 不再把 tool prompt 额外拼进 system prompt
- 也不再在普通工具调用后追加一段 runtime tool prompt

换句话说：

- `description`：工具的短介绍
- `guidance`：工具使用补充规则
- `prompt`：工具更完整的专属说明

这三者最终会统一折叠进工具 schema 的 `description` 字段发送给模型。

### 2. ToolResult 成为统一返回协议

工具不必只返回字符串。现在更推荐直接返回 `ToolResult`，以表达：

- 执行成功或失败
- 面向模型/用户的文本
- 结构化 JSON 数据
- 临时上下文
- 错误类型

### 3. Skill 通过现有 Tool 的临时授权完成能力展开

目录式 Skill 不动态导入 Python `tools.py`，也不创建隐式 Tool 实例。`allowed-tools` 只对 Agent 已注册的工具增加 invocation 级权限，并在 deferred schema 模式下展开对应 schema；Agent invoke 结束后临时权限和展开状态自动清理。

---

## 核心架构

```text
┌────────────────────────────────────────────┐
│                  BaseAgent                  │
│  ┌──────────────────────────────────────┐  │
│  │            ToolRegistry               │  │
│  │  register / mount / execute / clear   │  │
│  └──────────────────────────────────────┘  │
│                    ▲                        │
│                    │                        │
│      ┌─────────────┴─────────────┐         │
│      │                           │         │
│   resident tools            runtime tools   │
│      │                           │         │
│      ▼                           ▼         │
│   BasicAgent                skill_tool      │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│                   Tool                      │
│  ToolSpec + run() + ToolResult              │
└────────────────────────────────────────────┘
```

**关键组件**

| 组件 | 职责 |
|------|------|
| `Tool` | 工具抽象基类 |
| `ToolSpec` | 元数据对象，负责 schema / 描述 /属性表达 |
| `ToolResult` | 工具执行结果协议 |
| `ToolRegistry` | 工具注册、挂载、执行与可见性管理 |
| `BaseAgent` | 把 ToolRegistry 提供给 LLM，并执行工具调用循环 |
| `SkillManager` | 在 Skill 激活时注入或移除相关工具 |

---

## 快速开始

### 1. 定义一个最小自定义 Tool

```python
from pydantic import BaseModel, Field
from Tool.BaseTool import Tool


class WeatherParams(BaseModel):
    city: str = Field(description="城市名称")


class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="weather_tool",
            description="查询指定城市天气",
            parameters=WeatherParams,
        )

    def run(self, parameters: dict):
        city = parameters["city"]
        return f"{city} 今天晴，24°C"
```

### 2. 注册到 ToolRegistry

```python
from Tool.ToolRegistry import ToolRegistry

registry = ToolRegistry()
registry.register_tool(WeatherTool())
```

### 3. 接入 Agent

```python
from agent.BasicAgent import BasicAgent
from core.llm import EasyLLM

llm = EasyLLM(provider="openai", model="gpt-4o-mini")

agent = BasicAgent(
    name="assistant",
    llm=llm,
).with_tool(registry)

result = agent.invoke("帮我查一下北京天气")
```

### 4. 使用装饰器快速定义 Tool

```python
from pydantic import BaseModel, Field
from Tool.ToolRegistry import ToolRegistry

registry = ToolRegistry()


class EchoParams(BaseModel):
    text: str = Field(description="要回显的文本")


@registry.tool("echo_tool", "回显输入文本", EchoParams)
def echo_tool(text: str) -> str:
    return f"echo:{text}"
```

### 5. 返回结构化 ToolResult

```python
from Tool.BaseTool import Tool, ToolResult


class SearchTool(Tool):
    def run(self, parameters: dict):
        results = [
            {"title": "Example", "url": "https://example.com"},
        ]
        return ToolResult.success(
            content="找到 1 条结果",
            structured_data=results,
            metadata={"source": "demo"},
        )
```

---

## Tool 基类详解

所有工具都继承自 `Tool`：

```python
class Tool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Type[BaseModel],
        *,
        guidance: str = "",
        read_only: bool = False,
        destructive: bool = False,
        requires_confirmation: bool = False,
        supports_parallel: bool = True,
        output_mode: Literal["text", "json", "markdown"] = "text",
        source: str = "custom",
        ephemeral: bool = False,
        prompt: str = "",
        ...
    )
```

### 必须实现的方法

```python
def run(self, parameters: dict) -> Any:
    ...
```

`run()` 可以返回：

- `str`
- `dict`
- `list`
- `None`
- `ToolResult`

如果返回的是普通字符串或 JSON，`ToolRegistry` 会自动包装成 `ToolResult`。

### 常用辅助方法

| 方法 | 说明 |
|------|------|
| `validate_parameters()` | 用 Pydantic 模型校验参数 |
| `get_spec()` | 获取完整 `ToolSpec` |
| `get_openai_schema()` | 生成 OpenAI function calling schema |
| `to_provider_schema()` | provider 适配入口 |

---

## ToolSpec 说明

`ToolSpec` 是工具元数据对象。

### 关键字段

| 字段 | 说明 |
|------|------|
| `name` | 工具名 |
| `description` | 基础说明 |
| `parameters_model` | 参数模型（Pydantic） |
| `guidance` | 额外工具使用规则 |
| `read_only` | 是否只读 |
| `destructive` | 是否高风险/有副作用 |
| `requires_confirmation` | 是否需要确认 |
| `supports_parallel` | 是否适合并行调用 |
| `output_mode` | 主要输出模式 |
| `source` | 工具来源，例如 `builtin` / `custom` / `mcp` |
| `ephemeral` | 是否偏临时能力 |
| `prompt` | 工具的详细说明文本 |
| `expose_in_deferred` | 是否在 deferred schema 模式下默认可见 |
| `tags` | 标签 |
| `metadata` | 扩展元数据 |

### schema description 的构成

`ToolSpec.to_openai_schema()` 不会只发送 `description`。  
当前会把以下内容统一折叠为工具 schema 的 `description`：

1. `description`
2. `guidance`
3. `prompt`
4. 如果是按需 Skill 临时工具，再补一段“当前只在 tools 集合里临时存在”的说明

这正是当前推荐设计：  
**工具专属说明进 tool schema，而不是额外注入 system prompt。**

### 关于 `prompt_visibility`

`prompt_visibility` 仍然保留在 `ToolSpec` 中，但已经降级为**兼容字段**：

- 它曾用于控制 tool prompt 是 resident 还是 runtime 注入
- 当前框架主路径已经不再依赖它
- 新工具不需要再围绕它设计逻辑

---

## ToolResult 说明

`ToolResult` 是统一的工具执行结果协议。

```python
ToolResult(
    status="success",
    content="...",
    display_text="...",
    structured_data=...,
    ephemeral_context=...,
    error_type="...",
    metadata={...},
)
```

### 状态

| 状态 | 含义 |
|------|------|
| `success` | 执行成功 |
| `error` | 执行失败 |
| `needs_confirmation` | 需要用户确认 |

### 关键字段

| 字段 | 说明 |
|------|------|
| `content` | 主文本内容 |
| `display_text` | 优先展示给模型/用户的文本 |
| `structured_data` | 结构化结果 |
| `ephemeral_context` | 当前推理链可用的临时上下文 |
| `error_type` | 错误类型 |
| `metadata` | 扩展信息 |

### 推荐构造方式

```python
ToolResult.success(...)
ToolResult.error(...)
ToolResult.needs_confirmation(...)
```

### 自动补充的 metadata

通过 `ToolRegistry.execute_tool_result()` 执行工具时，框架会自动补：

- `tool_name`
- `tool_visibility`
- `tool_source`
- `side_effect_level`
- `resource_scope`
- `visibility_scope`

---

## ToolRegistry 管理器

`ToolRegistry` 是工具的注册、挂载、执行和可见性管理中心。

### 注册与挂载

```python
registry.register_tool(tool)       # 常驻 resident tool
registry.mount_runtime_tool(tool)  # 本轮/当前 invoke 期间可见
registry.mount_turn_tool(tool)     # 更短生命周期的 turn 级工具
```

### 清理

```python
registry.clear_runtime_tools()
registry.unregister_tool("tool_name")
```

### 执行

```python
result = registry.execute_tool_result("weather_tool", {"city": "北京"})
text = registry.execute_tool("weather_tool", {"city": "北京"})
```

区别：

- `execute_tool_result()` 返回 `ToolResult`
- `execute_tool()` 返回展示文本字符串

### 可见性

当前工具可见性分层：

| 可见性 | 说明 |
|------|------|
| `resident` | 常驻工具 |
| `runtime` | 当前 invoke 临时挂载 |
| `turn` | 更短生命周期的当前轮工具 |

相关查询：

```python
registry.get_visible_tools(scope="all")
registry.list_tool_specs(scope="runtime")
registry.get_tool_visibility("tool_name")
registry.has_tool("tool_name")
registry.get_tool_names()
```

---

## 与 Agent / Skill 的集成

### Agent 如何执行工具

`BaseAgent` / `BasicAgent` 工具执行流程大致是：

1. LLM 产生 tool call
2. Agent 解析 tool args
3. `ToolRegistry.execute_tool_result(...)`
4. 得到 `ToolResult`
5. 用 `result.to_display_string()` 回填到对话
6. 若 `ToolResult.ephemeral_context` 不为空，则追加临时上下文消息

### Skill 如何使用工具

Skill 的 `allowed-tools` 只能引用 ToolRegistry 中已经存在的工具。SkillManager 为这些工具增加临时权限规则并展开 deferred schema，不负责动态导入或卸载 Tool。完整 Skill 正文通过 MetaMessage 注入，invoke 结束后正文和临时权限一起回收。

### 与 Claude Code 的对齐

当前 EasyAgent Tool 模型已经遵循下面这条原则：

- tool 详细说明尽量通过 tool schema description 传递
- 不把 tool prompt 堆进 system prompt
- 不对普通工具调用做额外 runtime tool prompt 注入

这是当前与 Claude Code 最接近、也最稳定的做法。

---

## 内置工具

当前常见内置工具包括：

| 工具 | 说明 |
|------|------|
| `CalculatorTool` | 数学计算 |
| `WebSearchTool` | 网络搜索 |
| `MCPWrappedTool` | 远程 MCP 工具包装 |
| `MCPListResourcesTool` | 列出 MCP 资源 |
| `MCPReadResourceTool` | 读取 MCP 资源 |
| Memory tools | `add_memory_tool`、`search_memory_tool`、`update_memory_tool` 等 |
| Skill meta tools | `skill_tool`、`skill_discovery_tool`、`load_skill_tool`、`unload_skill_tool` |

### Memory Tool 特点

Memory tools 已原生返回 `ToolResult`，并带有 memory-specific guidance/prompt，适合：

- 写入长期记忆
- 搜索历史记忆
- 更新或删除过时记忆
- 记忆维护

### MCP Tool 特点

MCP 集成现在分成三层：

- 远程 MCP tools：包装成 `MCPWrappedTool`
- MCP resources：通过 `MCPListResourcesTool` / `MCPReadResourceTool` 暴露
- MCP prompts：由 MCP 模块显式读取，不自动映射成目录式 Skill

其中 Tool 层遵循以下规则：

- MCP 工具会把远程说明也折叠进 schema description
- MCP annotations 会进一步映射到 `ToolSpec`：
  - `readOnlyHint` -> `read_only`
  - `destructiveHint` -> `destructive`
  - `openWorldHint` -> `requires_confirmation` / metadata
  - `idempotentHint` -> `supports_parallel`
- 模型看到的是“当前真的暴露出来的远程工具/资源工具”
- 不会额外把 MCP tool prompt 再拼进 system prompt
- MCP prompt 与目录式 Skill 是两个独立扩展面，不做隐式注册

---

## 设计约束与最佳实践

### 1. 优先把工具规则写进 schema description

推荐：

- `description` 放一句清晰定义
- `guidance` 放短规则
- `prompt` 放更完整但仍然紧凑的说明

不推荐：

- 再额外把工具说明复制进 system prompt
- 再做独立的 runtime tool prompt 注入

### 2. 能返回 ToolResult 就不要只返回字符串

如果工具有以下需求，优先返回 `ToolResult`：

- 要附带结构化 JSON
- 要区分成功/失败/确认
- 要加 metadata
- 要提供 `ephemeral_context`

### 3. 参数模型要尽量精确

参数模型越明确，LLM 的 tool call 越稳定：

- 字段名短而准确
- `Field(description=...)` 描述清楚
- 必填/选填边界明确

### 4. 高风险工具要显式标记

对于会写文件、删数据、调用远程副作用接口的工具，建议显式设置：

```python
destructive=True
requires_confirmation=True
```

### 5. on-demand Skill Tool 不要当常驻工具使用

如果一个工具来自 `skill_tool`：

- 不要假定它跨 `invoke` 仍存在
- 下一轮使用前应确认它当前仍在 `tools` 集合中

---

## API 参考

### Tool

```python
class Tool(ABC):
    def run(self, parameters: dict) -> Any: ...
    def validate_parameters(self, parameters: dict) -> dict[str, Any]: ...
    def get_spec(self) -> ToolSpec: ...
    def get_openai_schema(self) -> dict[str, Any]: ...
    def to_provider_schema(self, provider: str = "openai") -> dict[str, Any]: ...
```

### ToolRegistry

```python
class ToolRegistry:
    def register_tool(self, tool: Tool, *, visibility: Literal["resident", "runtime", "turn"] = "resident"): ...
    def mount_runtime_tool(self, tool: Tool): ...
    def mount_turn_tool(self, tool: Tool): ...
    def clear_runtime_tools(self) -> None: ...
    def execute_tool_result(self, name: str, parameters: dict[str, Any]) -> ToolResult: ...
    def execute_tool(self, name: str, parameters: dict) -> str: ...
    def get_visible_tools(self, scope: str = "all") -> list[Tool]: ...
    def list_tool_specs(self, scope: str = "all") -> list[ToolSpec]: ...
    def get_tool_spec(self, name: str) -> ToolSpec | None: ...
    def get_tool_visibility(self, name: str) -> Literal["resident", "runtime", "turn"] | None: ...
```

### ToolResult

```python
ToolResult.success(...)
ToolResult.error(...)
ToolResult.needs_confirmation(...)
result.to_display_string()
```

---

## 小结

当前 EasyAgent 的 Tool 系统可以概括成一句话：

**工具说明走 schema，工具执行走 ToolResult，Skill 只对现有工具增加临时权限与 deferred schema 展开。**

如果你要继续扩展 Tool 模块，推荐优先沿着下面这条线推进：

1. 先定义清晰的参数模型
2. 再补 `description / guidance / prompt`
3. 优先返回 `ToolResult`
4. 根据实际副作用标记风险属性
5. 若与 Skill 集成，明确它是否属于临时 on-demand 工具
