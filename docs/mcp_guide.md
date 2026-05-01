# MCP Guide

MCP 模块让 EasyAgent 能把远程 MCP server 提供的工具、资源和 prompts 接入本地 Agent。对做平台型产品的人来说，这一层的意义很大，因为它解决的是：

- 如何把远程工具统一接入 `ToolRegistry`
- 如何把 MCP resource 暴露为可调用能力
- 如何把远程 prompt/skill 体系和本地 Skill 系统接起来
- 如何处理认证、策略和允许范围

如果你把 EasyAgent 当成“构建不同 Agent 产品的框架”，MCP 往往是接企业内部系统和外部平台能力的关键桥梁。

相关文档：

- [Tool System Guide](./tool_system_guide.md)
- [Skill System Guide](./skill_system_guide.md)
- [Permissions Guide](./permissions_guide.md)

## 1. 核心对象

MCP 相关的主要对象有：

- `MCPClient`
  - 具体的 MCP 客户端实现，用于连接 server。
- `MCPHub`
  - 管理多个 MCP server 的统一枢纽。
- `MCPToolManager`
  - 把 MCP server 暴露的能力包装成 EasyAgent Tool。
- `MCPWrappedTool`
  - 单个远程 MCP tool 的本地包装器。
- `MCPListResourcesTool`
  - 列出某个 server 的资源。
- `MCPReadResourceTool`
  - 读取某个 server 的资源正文。
- `MCPHubListResourcesTool`
  - 从 hub 维度列资源。
- `MCPHubReadResourceTool`
  - 从 hub 维度读资源。
- `register_mcp_tools(...)`
  - 把 MCP tools 注册进 `ToolRegistry`。
- `register_mcp_resource_hub_tools(...)`
  - 把 hub 级 resource tools 注册进 `ToolRegistry`。
- `MCPAuthConfig`
  - 认证配置。
- `MCPPolicyContext`
  - 策略上下文，用于决定允许访问哪些 server/capability。

## 2. MCP 层解决什么问题

直接把远程能力塞给模型会有几个现实问题：

- schema 是远程来的，需要转换成本地 Tool schema
- 远程能力有认证、策略、server 边界
- 远程 resources 和 remote prompts 不等于普通本地 Tool
- 不同 server 返回的 JSON schema、annotation 和 hint 需要归一化

MCP 模块的职责，就是把这些差异吸收掉，向 Agent 暴露统一的：

- Tool
- Resource tool
- Prompt-based skill

## 3. `MCPToolManager` 做什么

这是 MCP 集成的核心运行时对象。

它主要负责：

1. 从 MCP server 获取 tool 列表
2. 把远程 JSON schema 转成本地 Pydantic 参数模型
3. 把远程 annotations 归一化为本地风险/指导信息
4. 生成 `MCPWrappedTool`
5. 生成 resource tools
6. 把远程 prompt 映射到本地 `SkillRegistry`

推荐理解：

- `MCPClient` 负责“连上远程 server”
- `MCPToolManager` 负责“把远程能力变成 EasyAgent 能消费的对象”

## 4. MCP Tool 与普通 Tool 的差异

在使用方式上，MCP Tool 看起来像普通 Tool，但内部语义不同：

- 它的 schema 通常来自远程 server，而不是本地手写
- 它可能有远程副作用
- 它的 read-only / destructive / open-world / idempotent 提示来自 MCP annotations
- 它可能依赖认证上下文和策略上下文

所以产品上要特别注意：

- MCP Tool 不是天然安全的
- MCP Tool 的能力范围应该受 `MCPPolicyContext` 限制

## 5. Resource tools 是什么

MCP 不只有可执行 tool，还有 resource。

resource 更像：

- 文档
- 数据对象
- 只读资源
- 可列目录、可读正文的远程信息源

EasyAgent 把它们包装成两类工具：

- 列资源
- 读资源

这比直接让模型“猜一个 URI 然后读”更稳，因为流程被分成两步：

1. 先 list
2. 再 read

这与 deferred resource discovery 的理念一致。

## 6. Remote prompt / skill 是什么

有些 MCP server 不只是暴露 tool，还会暴露 prompt 模板。EasyAgent 的处理思路不是简单把 prompt 文本塞进 system，而是更接近 Skill：

- 远程 prompt 列表先做 listing
- 真要用时再展开 prompt body

这使它和本地 Skill 系统天然适配。

因此从产品视角看，MCP prompt 更像：

- “远程 skill”
- “远程能力模板”

## 7. 一次典型执行流程

下面用一个典型场景说明：

1. 应用启动时初始化 `MCPClient` 或 `MCPHub`
2. 创建 `MCPToolManager`
3. `register_mcp_tools(registry, ...)`
4. MCP server 的 tool schema 被转换成本地 Tool
5. Agent 运行时看到这些工具，按普通 Tool 方式调用
6. 真正执行时，`MCPWrappedTool` 把参数发给远程 MCP server
7. 若有 resources，模型可先 list 再 read
8. 若有 prompt skill，可映射到 SkillRegistry 做按需能力展开

对应的本地实际执行代码示例如下（以连接官方 SQLite MCP Server 为例）：

```python
import asyncio
from easyagent import BasicAgent, EasyLLM
from Tool.ToolRegistry import ToolRegistry
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from Tool.mcp.manager import MCPToolManager
from Tool.mcp.tools import register_mcp_tools, register_mcp_resource_tools

async def main():
    # 1. 应用启动时初始化 MCPClient 参数
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-sqlite", "--db-path", "test.db"]
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 建立通信并初始化
            await session.initialize()
            
            # 2. 创建 MCPToolManager
            manager = MCPToolManager(session)
            
            registry = ToolRegistry()
            
            # 3 & 4. 获取远端 tool schema，转换为本地 Tool 并注册
            await register_mcp_tools(registry, session)
            
            # 7. 若有 resources，注册对应的读写工具以便模型先 list 再 read
            register_mcp_resource_tools(registry, manager)

            # 5. 绑定到 Agent，Agent 即可像普通 Tool 一样识别到它们
            agent = BasicAgent(
                name="mcp_agent",
                llm=EasyLLM(),
                enable_tool=True,
                tool_registry=registry
            )
            
            # 6. 执行时，模型大喊“帮我查表”，本地的 MCPWrappedTool 便会通过 session 把参数发给远端执行
            print("开始对话...")
            await agent.astream_invoke("帮我看看数据库里有哪些表？")

if __name__ == "__main__":
    asyncio.run(main())
```

这说明 MCP 是“远程能力桥接层”，而不是另起一套 Agent 系统。

## 8. 如何接入 `BasicAgent`

常见最小接法：

```python
from easyagent import BasicAgent, EasyLLM
from easyagent.tools import register_mcp_tools
from Tool.ToolRegistry import ToolRegistry

registry = ToolRegistry()
my_mcp_client=
register_mcp_tools(registry, mcp_client=my_mcp_client)

agent = BasicAgent(
    name="assistant",
    llm=EasyLLM(),
    enable_tool=True,
    tool_registry=registry,
)
```

更完整的产品接法通常还会同时注入：

- `SkillRegistry`
- `PermissionContext`
- `MCPPolicyContext`

## 9. `MCPAuthConfig` 和 `MCPPolicyContext`

### `MCPAuthConfig`

用于描述如何和远程 MCP server 做认证。

典型作用：

- token
- headers
- 凭证来源
- 认证策略

### `MCPPolicyContext`

用于描述“当前 Agent 被允许访问哪些 MCP 能力”。

它适合承载：

- server allowlist
- capability allowlist
- 资源访问边界
- 只读/写入能力限制

从产品设计角度看：

- `MCPAuthConfig` 决定“能不能连”
- `MCPPolicyContext` 决定“连上后能做什么”

## 10. MCP 与权限系统的关系

MCP 本身不是权限系统，但它经常承载高风险能力，因此应该和 EasyAgent 的 permission engine 一起用。

推荐组合：

1. `MCPPolicyContext`
   - 限制 server/capability 范围
2. `PermissionContext`
   - 决定是否 ask / allow / deny
3. `requires_confirmation`
   - 对高风险远程能力做最终确认

也就是说，MCP 的风险控制通常是三层。

## 11. 适合的产品场景

很适合：

- 企业内部 agent 平台
- 连接 ticket / docs / DB / workflow 系统
- 需要把多个远程平台能力统一接入 Agent

不太适合：

- 只有一两个本地工具的小型 demo

## 12. 推荐实践

### 先做 capability 目录化

不要一开始就把所有远程能力直接全量暴露给模型。先有：

- 明确 server 划分
- 明确 capability 分类
- 明确 allowlist

### 优先把 resources 做成“两步走”

- 先 list
- 再 read

比直接盲读稳定得多。

### 让 MCP prompt 走 Skill 路径

远程 prompt 最好作为 on-demand skill 使用，而不是永久 system prompt。

### 策略和认证分开管理

这样更利于产品做多租户和多角色控制。

## 13. 常见坑

### 坑一：把 MCP 当成本地 Tool 完全等同处理

MCP Tool 的延迟、权限和失败模式都更复杂。

### 坑二：不加 policy 就全量暴露远程能力

这会让模型看到过宽的远程能力面，风险非常高。

### 坑三：把远程 prompt 直接拼进 system

这样会让 prompt 变重，并且不利于 cache 和按需展开。

### 坑四：跳过资源列表直接读资源

这样模型容易猜 URI，导致失败或读错资源。

### 坑五：认证放在 Tool metadata 里临时拼

认证配置应该有独立配置对象，而不是散落在单个工具定义中。
