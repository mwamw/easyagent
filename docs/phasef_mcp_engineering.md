# Phase F: MCP Engineering

## 本阶段完成了什么

Phase F 现在已经完成，MCP 不再只是“把远程 tool 包一层注册进 ToolRegistry”的轻量桥接，而是进入了框架的一等扩展面：

- 新增 `mcp/auth.py`
  - 统一 MCP 鉴权配置 `MCPAuthConfig`
  - 支持导出/恢复鉴权快照
- 新增 `mcp/policy.py`
  - `MCPPolicyContext / MCPPolicyRule / MCPPolicyDecision`
  - MCP server、capability kind、capability name 级别的策略控制
- 新增 `mcp/cache.py`
  - `MCPCapabilitySnapshot`
  - MCP capability/resource/prompt 的缓存与导出/恢复
- 新增 `mcp/connection_manager.py`
  - `MCPConnectionManager / MCPConnectionState`
  - 连接状态、错误分类、重试、策略检查
- `mcp/runtime.py`
  - `MCPRuntimeManager` 现在整合 connection manager、policy、cache、capability snapshot
  - 支持 `export_state()` / `restore_state()`
  - `MCPHub` 支持导出/恢复和统一 close
- `Tool/builtin/mcp_tool.py`
  - `MCPToolManager` 支持 `from_state()` / `export_state()`
  - 注册到 `ToolRegistry` 时会挂 runtime surface：`mcp_manager`、`mcp_hub`
  - MCP tool/resource tool 的 metadata 现在有统一 `source_identifier`
  - MCP 复杂 tool 的 guidance 明显更详细
  - MCP 错误结果现在有明确分类和更可读的错误说明
- `Tool/ToolRegistry.py`
  - 新增 runtime surface 挂点：
    - `register_runtime_surface()`
    - `get_runtime_surface()`
    - `list_runtime_surfaces()`
    - `unregister_runtime_surface()`
  - tool result metadata 会保留 `source_identifier`
- `runtime/context.py`
  - `ExecutionContext` 现在显式记录 `mcp_servers`
- `core/permissions/rules.py`
  - Permission matcher 现在支持 `matcher={"mcp_servers": [...]}`
- `core/agent.py`
  - session snapshot 新增 `mcp_runtime`
  - `load_session()` 支持 `mcp_client_overrides`
  - MCP runtime 会进入 restore report
  - `BaseAgent.close()` 会产出 `mcp_runtime` close report

## 现阶段框架的变换

在 Phase F 之前，EasyAgent 的 MCP 更像“远程工具适配层”：

- 能连上 server
- 能列 tool / 调 tool / 读 resource
- 能把 prompt 映射成 skill

但缺了框架真正需要的工程能力：

- 不知道连接现在是什么状态
- 不知道上次失败是什么类型
- 不知道 capability 快照和缓存怎么保存/恢复
- session restore 之后不会重建 MCP runtime
- `BaseAgent.close()` 不会收口 MCP
- 权限系统也不能按 MCP server 维度精确控制

现在这层已经变成正式 runtime：

- MCP server 有显式的连接状态与错误分类
- capability/resource/prompt 有缓存和快照
- ToolRegistry 里能挂 MCP runtime surface，而不只是工具实例
- session save/load 可以带上 MCP runtime
- close report 会明确反映 MCP runtime 的关闭结果
- permission rule 可以直接针对某个 MCP server

## 一个具体过程例子

下面是现在框架里一个真实而完整的 MCP 过程：

1. 创建 `ToolRegistry`
2. 创建 `MCPPolicyContext`
   - 例如允许自动连接
   - capability snapshot TTL 设为 300 秒
   - 禁止读取某个 prompt
3. 用 `register_mcp_tools(...)` 把本地 Python MCP server 注册进 registry
4. `MCPToolManager` 会：
   - 建立自己的 connection manager
   - 带上 policy/cache/auth
   - 把自身挂到 registry 的 `mcp_manager` runtime surface
5. Agent 运行时调用 MCP tool
   - 先过全局 permission engine
   - 再过 MCP policy
   - 再由 connection manager 执行远程操作并记录状态
6. 调用 `save_session()`
   - `mcp_runtime` 会一起进入 session snapshot
7. 调用 `load_session()`
   - registry 会自动重建 MCP runtime
   - restore report 里会出现 `components.mcp_runtime`
8. 调用 `agent.close()`
   - close report 里会出现 `components.mcp_runtime`

这意味着 MCP 已经不只是“本轮临时能调用一下”，而是一个可恢复、可观测、可治理的扩展面。

## 本阶段新增的关键接口

- `mcp.MCPAuthConfig`
- `mcp.MCPPolicyContext`
- `mcp.MCPPolicyRule`
- `mcp.MCPCapabilitySnapshot`
- `mcp.MCPServerCache`
- `mcp.MCPConnectionManager`
- `mcp.MCPConnectionState`
- `ToolRegistry.register_runtime_surface(...)`
- `BaseAgent.load_session(..., mcp_client_overrides=...)`

## 本阶段一个真实 example

真实 example 已放在：

- `example/example_phasef_mcp_engineering.py`

这个 example 使用真实 LLM：

```python
EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
```

流程包括：

- 注册本地 Python MCP server
- 建立 policy / capability snapshot
- 构造 BasicAgent
- 保存 session
- 重新 load session
- 查看 restore report
- 查看 close report

这个 example 没有被执行，保留给后续手动调试。

## 本阶段验证

我跑过：

```bash
python -m pytest test/test_mcp_tool.py test/test_mcp_engineering.py -q
python -m pytest test/test_session_persistence.py -k 'basic_agent_restores_mode_permissions_and_current_task or close_returns_cleanup_report or close_reports_degraded_background_runtime or basic_agent_session_restore_rebuilds_collaboration_runtime' -q
```

结果：

- `test_mcp_tool.py + test_mcp_engineering.py`: 全部通过
- 上述 session/close 相关回归：全部通过

## 下一步

按当前执行计划，Phase F 完成后，下一步进入 `Phase G：SDK 收口与通用 Agent 能力整理`。
