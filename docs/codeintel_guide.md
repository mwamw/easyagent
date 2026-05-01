# CodeIntel Guide

CodeIntel 模块为 Code Agent 提供比“全文 grep”更强的代码理解能力。它的核心价值在于：把“语言服务器 / 符号索引 / 工作区缓存”统一包装成 EasyAgent 可以直接使用的工具层和管理层。

如果你只是做普通聊天 Agent，这一层可以不用；但如果你做的是：

- IDE Agent
- 仓库审查 Agent
- 自动修复建议 Agent
- 大型仓库导航 Agent

那 CodeIntel 往往决定了体验上限。

相关文档：

- [Tool System Guide](./tool_system_guide.md)
- [Builtin Tools Catalog](./builtin_tools_catalog.md)
- [Worktree Guide](./worktree_guide.md)

## 1. 核心对象

CodeIntel 层最重要的对象有：

- `CodeIntelManager`
  - 总控入口，面向 Agent 提供统一查询接口。
- `CodeIntelProvider`
  - 抽象 provider 接口，定义“跳定义、查引用、查符号、查诊断”。
- `LSPCodeIntelProvider`
  - 基于语言服务器协议的 provider 实现。
- `WorkspaceCodeIntelCache`
  - 以工作区为单位缓存 codeintel 结果。
- 内置工具：
  - `CodeIntelStatus`
  - `FindDefinition`
  - `FindReferences`
  - `GetDocumentSymbols`
  - `GetWorkspaceSymbols`
  - `GetDiagnostics`
  - `CodeIntelCacheStatus`
  - `CodeIntelPrewarmWorkspace`

## 2. CodeIntel 解决什么问题

传统代码 Agent 只靠：

- `Grep`
- `Glob`
- `FileRead`

能完成很多任务，但一到大型仓库就会遇到这些问题：

- 定义跳转不准确
- 引用查找需要手写大量搜索规则
- 跨语言项目难做统一定位
- 诊断信息和语言服务器状态拿不到

CodeIntel 的目标是把这些能力变成统一接口。

## 3. `CodeIntelProvider` 的职责

这是抽象接口，定义 provider 至少要支持的查询类型：

- `find_definition`
- `find_references`
- `get_document_symbols`
- `get_workspace_symbols`
- `get_diagnostics`

因此 EasyAgent 的 code agent 可以只依赖抽象接口，而不是写死某个 LSP 实现。

这对于框架很重要，因为不同产品可能会使用：

- 本地 LSP
- 远程索引服务
- 自研符号引擎

## 4. `CodeIntelManager` 负责什么

`CodeIntelManager` 是真正面向 Agent 的运行时总控。

它负责：

1. 维护当前 `provider`
2. 推断 `workspace_root`
3. 推断 `allowed_roots`
4. 处理相对路径解析和边界校验
5. 维护每个 workspace 的 `WorkspaceCodeIntelCache`
6. 对外暴露统一查询方法

你可以把它理解成：

- provider 负责“怎么查”
- manager 负责“在当前工作区里安全、统一地查”

## 5. `CodeIntelManager` 的主要初始化参数

最常用参数有：

- `provider`
  - 必填。真正执行 codeintel 查询的 provider。
- `parent_agent`
  - 可选。若传入，manager 会尝试从 agent 的 `execution_context`、`config` 中推断工作区。
- `workspace_root`
  - 当前工作区根目录。
- `allowed_roots`
  - 允许访问的根目录集合。是工作区边界控制的重要部分。
- `cache`
  - 单个 `WorkspaceCodeIntelCache`。
- `workspace_caches`
  - 多工作区缓存集合。

如果你没有显式传 `workspace_root`，manager 会尽量从：

- `execution_context.worktree_path`
- `execution_context.workspace_root`
- `config.workspace_root`
- 当前进程 cwd

中推断。

## 6. `WorkspaceCodeIntelCache` 是什么

CodeIntel 查询通常比较贵，尤其在大型仓库和 LSP 启动初期。

`WorkspaceCodeIntelCache` 的目标是：

- 以工作区为单位缓存结果
- 避免重复做相同查询
- 让不同 worktree / workspace 各自维护独立缓存

这点和 `WorktreeManager` 的关系很紧密：如果你让子 agent 在独立 worktree 中工作，最好给它独立的 workspace cache 视图。

## 7. 内置 CodeIntel 工具分别做什么

### `CodeIntelStatus`

查看当前 codeintel provider 的可用状态。

适合：

- 启动自检
- 调试 provider 是否正常

### `FindDefinition`

根据：

- `file_path`
- `line`
- `column`

跳到符号定义。

### `FindReferences`

查一个符号在项目中的引用。

### `GetDocumentSymbols`

列出单文件中的符号树。很适合先快速理解一个文件结构。

### `GetWorkspaceSymbols`

在整个工作区按名字搜索符号。

### `GetDiagnostics`

读取文件的语言服务器诊断结果。

### `CodeIntelCacheStatus`

查看当前 workspace cache 的状态。

### `CodeIntelPrewarmWorkspace`

预热工作区索引或缓存，适合大仓库启动时做准备动作。

## 8. 一次典型执行流程

以“跳定义”举例：

1. 模型决定调用 `FindDefinition`
2. tool 把参数传给 `CodeIntelManager`
3. manager 先解析：
   - `workspace_root`
   - `allowed_roots`
   - `file_path`
4. manager 查询当前 workspace 对应的 cache
5. 如果缓存命中，直接返回
6. 如果未命中，调用 provider
7. provider 执行真实的 LSP/索引查询
8. 结果被包装为统一 `CodeIntelQueryResult`
9. manager 更新 cache 并返回给 tool

所以内置 tool 不是直接调用 LSP，而是通过 manager 做了一层工作区安全和缓存控制。

## 9. 如何接入 `BasicAgent`

最常见接法：

```python
from easyagent import BasicAgent, EasyLLM
from easyagent.tools import register_codeintel_tools
from codeintel.lsp import LSPCodeIntelProvider
from codeintel.manager import CodeIntelManager
from Tool.ToolRegistry import ToolRegistry

provider = LSPCodeIntelProvider(...)
manager = CodeIntelManager(provider=provider, workspace_root=".")

registry = ToolRegistry()
register_codeintel_tools(registry, manager=manager)

agent = BasicAgent(
    name="code-agent",
    llm=EasyLLM(),
    enable_tool=True,
    tool_registry=registry,
    codeintel_manager=manager,
)
```

如果你的 Agent 有 `execution_context` 或 worktree，manager 可以自动跟随当前工作区。

## 10. 和 Worktree 的关系

Code agent 很常见的场景是：

- 主 agent 在主工作树
- 子 agent 在 worktree 中做实验性修改

这时最重要的是不要混用工作区边界和缓存：

- 每个 worktree 应视为独立 workspace
- `workspace_root` 应跟随当前 `execution_context.worktree_path`
- codeintel cache 最好按 workspace 拆开

## 11. 推荐的产品使用方式

### 模式一：只读分析型 code agent

注册：

- `FindDefinition`
- `FindReferences`
- `GetDocumentSymbols`
- `GetWorkspaceSymbols`
- `GetDiagnostics`

适合：

- repo review
- code explanation
- architecture audit

### 模式二：IDE 辅助型 agent

额外加入：

- `CodeIntelStatus`
- `CodeIntelCacheStatus`
- `CodeIntelPrewarmWorkspace`

适合：

- 启动自检
- 预热大型仓库

## 12. 常见坑

### 坑一：把 codeintel 当成文件搜索替代品

CodeIntel 很强，但不是所有问题都适合走它。很多全局字符串搜索仍然应该先用 `Grep`。

### 坑二：不设置 `allowed_roots`

这会让文件解析边界变模糊，尤其在多 workspace 或 worktree 下容易出问题。

### 坑三：多个 worktree 共享同一套 workspace 视图

会导致定位结果和缓存错乱。

### 坑四：把 provider 和 manager 写死在一起

框架层应该尽量依赖 `CodeIntelProvider` 抽象，而不是直接耦合某个具体 LSP 实现。

### 坑五：不做预热就直接在大仓库里高频调用

首次体验会明显变差。大仓库更适合启动时做预热。
