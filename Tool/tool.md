# EasyAgent Tool 模块文档

`EasyAgent` 的 Tool 模块是一个高度解耦、可扩展的工具调用系统。它不仅为大模型（LLM）提供了与外部世界（文件系统、终端、互联网、MCP 服务）交互的能力，还内置了完善的安全沙盒、结果截断、以及统一的执行协议。

---

## 1. 核心架构

Tool 模块由四个关键组件构成：

### 1.1 `Tool` (基类)
所有工具必须继承自 `EasyAgent.Tool.BaseTool.Tool`。它定义了工具的执行逻辑和元数据。
- `run(parameters)`: 同步执行逻辑。
- `arun(parameters)`: 异步执行逻辑（可选，默认回退到 `run`）。
- `parameters`: 一个 Pydantic `BaseModel` 类，用于定义和校验 LLM 传入的参数。

### 1.2 `ToolSpec` (元数据)
描述工具的静态属性，包括：
- `name` & `description`: 发送给 LLM 的名称和描述。
- `read_only`: 标识工具是否只读。
- `destructive`: 标识工具是否具有破坏性。
- `requires_confirmation`: 标识工具执行前是否需要人类确认。
- `output_mode`: 返回格式（text/json/markdown）。

### 1.3 `ToolResult` (标准输出)
工具执行后的统一返回对象。
- `status`: "success" / "error" / "needs_confirmation"。
- `content`: 给 LLM 阅读的原始文本。
- `display_text`: 在 UI 界面展示的友好文本。
- `structured_data`: 包含完整执行结果的结构化字典（UI 或后续处理使用）。

### 1.4 `ToolRegistry` (注册表)
负责工具的生命周期管理、权限校验和分发执行。它支持将工具注册为：
- `resident`: 常驻工具（全局可用）。
- `runtime`: 运行时工具（会话级别）。
- `turn`: 回合工具（单次请求可用）。

---

## 2. 快速开始：定义一个工具

### 2.1 类定义方式 (推荐)
适合逻辑复杂、需要状态管理的工具。

```python
from pydantic import BaseModel, Field
from EasyAgent.Tool.BaseTool import Tool, ToolResult

# 1. 定义参数模型
class MyToolParams(BaseModel):
    query: str = Field(description="搜索关键词")

# 2. 实现工具类
class MyTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="我的自定义工具描述",
            parameters=MyToolParams,
            read_only=True
        )

    def run(self, parameters: dict) -> ToolResult:
        query = parameters["query"]
        # 执行逻辑...
        return ToolResult.success(f"找到了关于 {query} 的结果")
```

### 2.2 装饰器方式
适合快速封装纯函数。

```python
from EasyAgent.Tool.ToolRegistry import ToolRegistry
from pydantic import BaseModel

registry = ToolRegistry()

class AddParams(BaseModel):
    a: int
    b: int

@registry.tool(name="add", description="计算两个数的和", parameters=AddParams)
def add_func(a: int, b: int):
    return a + b
```

---

## 3. 内置工具库详解

`EasyAgent` 预置了丰富的内置工具，位于 `EasyAgent.Tool.builtin` 目录下。

### 3.1 文件系统工具 (`filesystem.py`, `file_edit.py`, `file_write.py`)
提供了受限且安全的文件操作能力。
- **`FileReadTool`**: 读取文件内容（支持 `.pdf` 和文本）。支持 `offset` 和 `limit` 分页读取大文件。
- **`GlobTool`**: 使用通配符（如 `src/**/*.py`）递归查找文件。
- **`GrepTool`**: 高性能正则搜索（支持 `ripgrep` 降级到纯 Python 实现）。
- **`FileEditTool`**: 针对代码的增量编辑。支持“精确匹配”和“规范化模糊匹配”两级容错。
- **`FileWriteTool`**: 原子化文件写入，防止写入中断导致文件损坏。

### 3.2 命令行工具 (`bash_tool.py`)
- **`BashTool`**: 在本地终端执行 Shell 命令。支持后台运行（`run_in_background`）和标准输出/错误流捕获。
- **`TaskOutputTool`**: 获取后台运行任务的实时输出或轮询状态。
- **`TaskStopTool`**: 停止正在运行的后台任务。

### 3.3 网络与搜索工具 (`search.py`, `web_fetch.py`)
- **`WebSearchTool`**: 在互联网上搜索信息（支持 SerpAPI 和 DuckDuckGo 后端）。
- **`WebFetchTool`**: 抓取特定 URL 的正文。内置启发式算法，可根据 `prompt` 自动提取网页中最相关的段落。

### 3.4 记忆管理工具 (`memorytool.py`)
- **`AddMemoryTool` / `SearchMemoryTool`**: 允许 Agent 自主维护长期知识。支持四种记忆类型：`working`（任务约束）、`episodic`（经历）、`semantic`（事实）、`perceptual`（多模态感知）。

### 3.5 实用工具
- **`CalculatorTool`**: 安全的数学表达式求值器，基于 AST 解析，防止代码注入。
- **`TodoWriteTool`**: 维护任务进度清单（TodoList），对齐 Claude Code 原生体验。

### 3.6 MCP 工具 (`mcp_tool.py`)
- **`MCPHub`**: 统一管理 Model Context Protocol (MCP) 服务。允许动态加载来自外部服务器的工具和资源。

---

## 4. 关键特性说明

### 4.1 安全防呆：先读后写 (Read-before-Write)
`FileEdit` 和 `FileWrite` 工具强制执行“先读后写”原则。如果 Agent 尝试修改一个它从未读取过（或者自上次读取后已被修改）的文件，工具会报错并要求 Agent 重新读取。这有效防止了模型因过时信息而破坏代码。

### 4.2 智能结果截断
所有文件读取和 Shell 输出工具都内置了 `_clip_text` 逻辑。当输出超过 `DEFAULT_MAX_OUTPUT_CHARS`（默认约 12 万字）时，会自动截断并附加提示，防止 LLM 上下文窗口爆炸。

### 4.3 原子化操作
`FileWriteTool` 使用临时文件 + `os.replace` 的方式实现原子写入。即便系统在写入过程中崩溃，原始文件也会保持完整。

### 4.4 多模型适配
通过 `Tool.to_provider_schema()`，工具定义可以自动转换为 OpenAI 风格的 Function Calling 或 Claude 风格的 Tool Use 定义。

---

## 5. 注册与使用示例

```python
from EasyAgent.Tool.ToolRegistry import ToolRegistry
from EasyAgent.Tool.builtin import register_filesystem_tools, register_shell_tools

registry = ToolRegistry()

# 一键注册核心编码工具
workspace = "/path/to/your/project"
register_filesystem_tools(registry, workspace_root=workspace)
register_shell_tools(registry, workspace_root=workspace)

# LLM 调用示例
tools_for_llm = registry.get_openai_tools()
# ... 发送 tools_for_llm 给模型 ...

# 执行模型返回的调用
result = registry.execute_tool("FileRead", {"path": "README.md"})
print(result) # 输出文件内容文本
```
