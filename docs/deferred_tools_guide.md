# EasyAgent Deferred Tools Guide

这份文档专门解释 `tool_schema_mode="deferred"` 的完整行为、默认规则和自定义方式。

---

## 1. 这套机制解决什么问题

如果一个 agent 同时挂了很多工具，直接把所有 schema 一次性发给模型会带来几个问题：

- 初始 `tools` payload 很大
- prompt cache 前缀更容易抖动
- 明明只会用到 2 个工具，却要把几十个 schema 全量送进去

`deferred` 的目标就是把工具系统拆成两层：

1. 工具目录
2. 按需展开的完整 schema

---

## 2. 基本行为

当：

```python
config = Config(tool_schema_mode="deferred")
```

时，一个 invoke 会这样工作：

1. 初始请求不发送所有完整 tool schema
2. 初始 `tools` 集合只包含：
   - `tool_schema_tool`
   - 默认暴露的工具
   - 当前 invoke 已经展开过的工具
3. 如果模型要用某个当前未展开的工具，它先调用：

```text
tool_schema_tool
```

4. `tool_schema_tool` 会把指定工具加入当前 invoke 的 expanded set
5. 下一轮同一个 invoke 请求里，该工具 schema 才真正进入 provider `tools` payload
6. invoke 结束后，expanded set 自动清空

---

## 3. 什么叫“默认暴露”

默认暴露的意思是：

> 在 deferred 模式下，这个工具不用先经过 `tool_schema_tool` 展开，也会直接出现在初始 `tools` 集合里。

控制它的正式字段只有一个：

```python
Tool(..., expose_in_deferred=True)
```

以及注册时的覆盖参数：

```python
registry.register_tool(tool, expose_in_deferred=True)
```

---

## 4. 内置工具现在怎么工作

当前规则是：

- 各内置 `register_*_tool(...)` / `register_*_tools(...)` helper 默认 `expose_in_deferred=True`
- 你可以在注册时覆盖成 `False`
- 框架不再通过 `metadata["always_exposed"]` 或 `source == "builtin"` 这类隐式规则猜测

例如：

```python
register_filesystem_tools(registry, workspace_root="/repo")
register_shell_tools(registry, workspace_root="/repo", expose_in_deferred=False)
register_search_tool(registry, expose_in_deferred=True)
```

这意味着：

- `filesystem` 相关工具默认直接暴露
- `shell` 工具会被延迟展开
- `search` 工具直接暴露

---

## 5. 自定义工具怎么控制

### 方式一：在 Tool 构造时声明默认值

```python
tool = MyTool(expose_in_deferred=True)
registry.register_tool(tool)
```

### 方式二：注册时覆盖

```python
registry.register_tool(MyTool(), expose_in_deferred=True)
registry.register_tool(MyOtherTool(), expose_in_deferred=False)
```

优先建议：

- 工具作者在工具定义里给出合理默认值
- 产品层在注册时按场景覆盖

---

## 6. 推荐哪些工具默认暴露

通常建议默认暴露：

- 项目结构探索工具
  - `FileRead`
  - `List`
  - `Glob`
  - `Grep`
- 轻量只读分析工具
- 高频基础协调工具
  - `AskUserQuestion`
  - `TodoWrite`
  - `TaskGet`
  - `TaskList`

通常不建议默认暴露：

- 高风险写入工具
  - `Bash`
  - `FileWrite`
  - `FileEdit`
- 低频重型工具
- 只在少数任务中使用的专用工具

---

## 7. 一个完整例子

```python
from easyagent import BasicAgent, EasyLLM, Config
from easyagent.tools import (
    ToolRegistry,
    register_filesystem_tools,
    register_file_edit_tool,
    register_file_write_tool,
    register_shell_tools,
)

llm = EasyLLM(provider="anthropic_native", model="deepseek-v4-flash:zenmux:claude")
config = Config(tool_schema_mode="deferred")

registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root="/repo", expose_in_deferred=True)
register_file_edit_tool(registry, workspace_root="/repo", expose_in_deferred=False)
register_file_write_tool(registry, workspace_root="/repo", expose_in_deferred=False)
register_shell_tools(registry, workspace_root="/repo", expose_in_deferred=False)

agent = BasicAgent(name="code-agent", llm=llm, config=config).with_tool(registry)
```

此时：

- 初始请求就能直接使用 `List/Grep/FileRead/Glob`
- 但如果模型想调用 `FileEdit` 或 `Bash`，必须先走 `tool_schema_tool`

---

## 8. 和 prompt cache 的关系

deferred tools 的价值不只是“省 token”，更是“稳前缀”：

- 常用基础工具稳定地留在前缀里
- 重型/低频工具只在需要时展开
- 新增一个低频工具，不会让所有请求的初始 tools payload 都变化

所以它是 cache 架构的一部分，不只是工具体验优化。

---

## 9. 相关源码

- [Tool/BaseTool.py](../Tool/BaseTool.py)
- [Tool/ToolRegistry.py](../Tool/ToolRegistry.py)
- [Tool/builtin/tool_schema_tool.py](../Tool/builtin/tool_schema_tool.py)
- [Tool Authoring Guide](./tool_authoring_guide.md)
