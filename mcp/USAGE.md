# MCP 模块使用说明（真实服务示例）

本文档用真实 MCP 服务演示 EasyAgent 的 MCP 系统如何接入与调用。

## 1. 依赖准备

```bash
pip install fastmcp>=2.0.0
```

如果要运行 npx 示例，还需要：

```bash
node -v
npx -v
```

## 2. 目录中的示例文件

- `mcp/examples/real_python_mcp_server.py`
    - 真实 Python MCP Server（stdio）
- `mcp/examples/real_python_mcp_server_text.py`
    - 真实 Python 文本工具 MCP Server（stdio）
- `mcp/examples/real_python_mcp_server_structured.py`
    - 真实 Python 结构化工具 MCP Server（stdio）
- `mcp/examples/example_real_python_stdio_client.py`
    - 真实 Python 客户端连接示例（连上面的 server）
- `mcp/examples/example_real_npx_filesystem_client.py`
    - 真实 npx MCP Server 连接示例（filesystem server）

在 `test/test_mcp_real_integration.py` 中还包含了更多真实服务探测：

1. Python stdio 服务（基础）
2. Python stdio 服务（文本工具）
3. Python stdio 服务（结构化工具）
4. npx filesystem 服务
5. npx memory 服务（环境可用时）
6. npx time 服务（环境可用时）

## 3. 示例 A：Python 程序 MCP 服务（stdio）

### 3.1 启动并调用（推荐直接跑客户端示例）

从项目根目录运行：

```bash
python mcp/examples/example_real_python_stdio_client.py
```

这个示例会（全部基于真实服务）：

1. 通过 `MCPClient` 连接真实 Python MCP Server
2. 调用 `list_tools` 获取真实工具列表
3. 调用 `echo`、`add`、`repeat`
4. 结束时关闭连接

### 3.2 服务端脚本（可单独查看）

```bash
python mcp/examples/real_python_mcp_server.py
```

它基于 `mcp.mcp_server.MCPServer` 暴露了三个真实工具：

1. `echo`
2. `add`
3. `repeat`

## 4. 示例 B：npx MCP 服务（filesystem）

从项目根目录运行：

```bash
python mcp/examples/example_real_npx_filesystem_client.py
```

这个示例会（全部基于真实服务）：

1. 通过 `npx -y @modelcontextprotocol/server-filesystem <workspace>` 启动真实服务
2. 用 `MCPClient` 拉取远程工具列表并打印
3. 尝试调用常见文件系统工具（如 `list_directory` / `read_file`）
4. 结束时关闭连接

## 5. 在 Agent 中接入真实 MCP 工具

```python
from core.llm import EasyLLM
from agent.BasicAgent import BasicAgent
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin import register_mcp_tools

llm = EasyLLM()
registry = ToolRegistry()

manager = register_mcp_tools(
        registry=registry,
        server_source=["python", "mcp/examples/real_python_mcp_server.py"],
        tool_prefix="py_",
)

agent = BasicAgent(
        name="mcp-agent",
        llm=llm,
        enable_tool=True,
        tool_registry=registry,
)

print(agent.invoke("请调用 py_add 计算 12 + 30"))
manager.close()
```

说明：

- 上面是 Agent 场景（工具注册到 `ToolRegistry`）
- 目录里的 `mcp/examples/*.py` 是最直接的真实连接示例（更适合理解 MCP 协议交互）

## 6. 关键参数说明

- `server_source`
    - 可以是 Python 命令数组、npx 命令数组、URL、FastMCP 实例等
- `tool_prefix`
    - 给远程工具加前缀，避免和本地工具重名
- `auto_connect`
    - 默认 `True`，首次调用时自动连接

## 7. 常见问题

### 7.1 报错：Client not connected

如果你手动设置了 `auto_connect=False`，先调用：

```python
manager.connect()
```

### 7.2 npx 示例无法运行

常见原因：

1. 未安装 Node.js / npx
2. Node.js 版本过低（`@modelcontextprotocol/server-filesystem` 通常要求 Node >= 18）
3. 网络不可达，首次拉取 npm 包失败
4. 目标路径权限不足

### 7.3 工具参数校验失败

远程工具 schema 的必填字段没传齐。先打印工具定义再调用。
