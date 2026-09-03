# Code Agent Product Quickstart

这份文档演示如何用 EasyAgent 组装一个真正可用的 Code Agent 产品，而不是只做一个能调用模型的 demo。

相关文档：

- [README](../README.md)
- [Tool System Guide](./tool_system_guide.md)
- [Builtin Tools Catalog](./builtin_tools_catalog.md)
- [Permissions Guide](./permissions_guide.md)
- [Worktree Guide](./worktree_guide.md)

## 1. 一个推荐的最小产品骨架

```python
from easyagent import (
    BasicAgent,
    CallbackManager,
    Config,
    EasyLLM,
    InMemoryObservabilityStore,
    PermissionContext,
    PermissionEngine,
    StreamingCallback,
    TaskService,
)
from easyagent.tools import (
    ToolRegistry,
    register_filesystem_tools,
    register_shell_tools,
)

llm = EasyLLM(
    provider="anthropic_native",
    base_url="http://127.0.0.1:5124",
    api_key="x",
    model="deepseek-v4-flash:zenmux:claude",
)

config = Config(
    tool_schema_mode="deferred",
    workspace_root=".",
    allowed_roots=["."],
    enable_worktree=True,
)

registry = ToolRegistry()
register_filesystem_tools(registry, workspace_root=".")
register_shell_tools(registry, workspace_root=".", expose_in_deferred=False)
agent = BasicAgent(
    name="code-agent",
    llm=llm,
    config=config,
)
agent.with_tool(registry)
agent.with_permissions(PermissionEngine(), PermissionContext())
agent.with_callbacks(CallbackManager([StreamingCallback()]))
agent.with_task_service(TaskService())
agent.with_codeintel()
agent.with_worktree()
agent.with_multi_agent(workspace_root=".")
agent.with_observability(store=InMemoryObservabilityStore())
```

## 2. 推荐的工具组合

### 默认直接暴露

- 文件系统探索工具
- task / todo 工具
- 轻量 codeintel
- 必要时的 agent runtime 工具

### 推荐 deferred 展开

- `Bash`
- `FileEdit`
- `FileWrite`
- 一次性低频重型工具

## 3. Ask / Allow / Deny

Code Agent 几乎总要接权限系统。  
否则模型能不能直接改文件、执行 shell、发消息，边界会非常模糊。

详见：

- [Permissions Guide](./permissions_guide.md)

## 4. Worktree 与子 Agent

如果你要让 agent：

- 并发分析不同代码面
- 在隔离环境里做修改
- 后台跑审计 / 测试 / diff

那就应接：

- `AgentRuntimeManager`
- `WorktreeManager`

详见：

- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)
- [Worktree Guide](./worktree_guide.md)

## 5. 推荐的产品级补充

至少再接：

- callback
- observability recorder
- session store
- codeintel

进阶再接：

- skill system
- memory
- MCP

## 6. 你最终得到的是什么

如果按上面的方式装配，最终的产品不是“一个会聊天的模型”，而是：

- 有稳定工具系统
- 有权限确认
- 有 deferred tool schema
- 有 runtime / subagent / task 能力
- 有 cache 观测
- 有 session / restore 能力
- 可以继续扩展成更完整 IDE / CLI / Web agent 的基础框架
