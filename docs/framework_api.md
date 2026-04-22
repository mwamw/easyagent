# EasyAgent Framework API

这份文档定义当前推荐的公共 SDK 入口。

## 推荐导入方式

外部项目现在应优先从 `easyagent` 导入，而不是直接依赖内部目录：

```python
from easyagent import BasicAgent, EasyLLM, ToolRegistry
from easyagent.mcp import register_mcp_tools
from easyagent.observability import InMemoryObservabilityRecorder
from easyagent.permissions import PermissionContext, PermissionRule, PermissionBehavior
from easyagent.tasks import TaskService
```

## 顶层常用导出

`easyagent` 顶层适合导入最常用对象：

- Agent
  - `BasicAgent`
  - `ConversationalAgent`
  - `PlanningAgent`
  - `ReactAgent`
  - `StructuredOutputAgent`
- LLM
  - `EasyLLM`
- Tool
  - `Tool`
  - `ToolRegistry`
  - `ToolResult`
  - `ToolSpec`
- Task
  - `TaskRecord`
  - `TaskStatus`
  - `TaskService`
- Permission
  - `PermissionContext`
  - `PermissionRule`
  - `PermissionBehavior`
  - `PermissionMode`
  - `PermissionEngine`
- Session
  - `SessionStore`
  - `ConversationStore`
  - `SessionRestoreReport`
- Runtime
  - `ExecutionContext`
  - `TeamManager`
- MCP
  - `register_mcp_tools`
  - `MCPToolManager`
  - `MCPPolicyContext`
  - `MCPAuthConfig`
- Observability
  - `BaseObservabilityRecorder`
  - `InMemoryObservabilityRecorder`

## 分类子模块

如果你需要更清晰的模块边界，使用这些子模块：

- `easyagent.agents`
- `easyagent.llms`
- `easyagent.tools`
- `easyagent.tasks`
- `easyagent.permissions`
- `easyagent.session`
- `easyagent.runtime`
- `easyagent.mcp`
- `easyagent.hooks`
- `easyagent.guardrails`
- `easyagent.skills`
- `easyagent.context`
- `easyagent.codeintel`
- `easyagent.observability`
- `easyagent.rag`
- `easyagent.memory`

## 兼容说明

当前仓库仍然保留旧路径：

- `agent`
- `core`
- `Tool`
- `runtime`
- `task`
- `mcp`
- 其他内部目录

这些路径目前不会立即删除，但不建议新的外部项目继续把它们当作稳定 SDK 边界。

## 安装方式

核心安装：

```bash
pip install -e .
```

可选扩展：

```bash
pip install -e ".[mcp]"
pip install -e ".[rag]"
pip install -e ".[memory]"
pip install -e ".[dev]"
```

## 一个最小 SDK 示例

```python
from easyagent import BasicAgent, EasyLLM

llm = EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)

agent = BasicAgent(name="assistant", llm=llm)
response = agent.invoke("用一句话说明 EasyAgent 现在是什么。")
print(response)

summary = agent.get_observability_summary()
print(summary)
```

## 当前边界原则

Phase G 之后，框架对外的稳定边界是：

- `easyagent` 作为公共入口
- `pyproject.toml` 作为安装与 extras 定义
- `docs/framework_api.md` 与 `example/README.md` 作为公共使用索引

内部目录结构仍可能继续演进，但应尽量通过 `easyagent.*` 保持兼容。
