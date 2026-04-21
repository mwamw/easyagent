# Phase D Code Intelligence v1

本文档记录本轮完成的能力：把 EasyAgent 从“只有文件级读取/搜索工具”升级成“具备可插拔 code intelligence/LSP 子系统”的框架。

## 本轮完成了什么

### 1. `codeintel/` 正式落地

新增：

- `codeintel/models.py`
- `codeintel/provider.py`
- `codeintel/manager.py`
- `codeintel/lsp/client.py`
- `codeintel/lsp/provider.py`

现在框架里已经有一套正式的 codeintel 协议：

- `CodeIntelProvider`
- `CodeIntelManager`
- `DefinitionQuery`
- `ReferenceQuery`
- `DocumentSymbolsQuery`
- `WorkspaceSymbolsQuery`
- `DiagnosticsQuery`
- `CodeIntelQueryResult`

这意味着之后无论接 LSP、离线索引器还是别的 provider，都不需要直接改工具层接口。

### 2. LSP stdio provider 已可用

本轮不是只加了抽象，还落了最小可用的 stdio LSP client：

- 启动语言服务器
- `initialize / initialized`
- `textDocument/didOpen`
- `definition`
- `references`
- `documentSymbol`
- `workspace/symbol`
- `publishDiagnostics`
- `shutdown / exit`

默认会按文件扩展名尝试常见 server：

- Python: `basedpyright-langserver` / `pyright-langserver` / `pylsp` / `jedi-language-server`
- TypeScript / JavaScript: `typescript-language-server`
- Rust: `rust-analyzer`
- Go: `gopls`
- Java: `jdtls`
- C/C++: `clangd`

如果你希望显式指定 server，也可以自己构造 `LSPCodeIntelProvider(server_command=[...])`。

### 3. Codeintel 工具已正式暴露

新增工具：

- `CodeIntelStatus`
- `FindDefinition`
- `FindReferences`
- `GetDocumentSymbols`
- `GetWorkspaceSymbols`
- `GetDiagnostics`

注册入口：

- `register_codeintel_tools(...)`

这些工具不是简单把 LSP 原始响应原样塞给模型，而是统一返回：

- `status`
- `providerName`
- `workspaceRoot`
- `items`
- `fallbackTools`
- `metadata`

并且会把结果放进 `ToolResult.ephemeral_context`，方便后续推理链继续消费。

### 4. fallback 语义正式落地

本轮把 `codeintel 可用` 和 `codeintel 不可用` 都做成了正式协议。

当 LSP server 不可用时，工具不会只抛一个模糊错误，而是会返回结构化结果：

- `status = unavailable`
- `fallbackTools = ["FileRead", "Grep", "Glob"]`
- 明确说明不可用原因

这意味着 agent 可以稳定退回文件级分析路径，而不是继续无意义重试。

### 5. 复杂工具提示词已补细

本轮 codeintel 工具的 prompt/guidance 都按“复杂函数必须写清楚”的标准补齐了。

例如：

- `FindDefinition` 明确要求坐标必须落在标识符上，且只适合已知精确引用点
- `FindReferences` 明确说明适合做影响面分析，并提醒不要把全部命中都读一遍
- `GetDocumentSymbols` 明确说明它适合大文件结构摸排
- `GetDiagnostics` 明确说明 diagnostics 为空不等于绝对正确

这一步的目的不是“描述更长”，而是让模型知道：

- 什么时候该用哪个工具
- 什么时候该停止使用 codeintel
- 什么时候该退回 `Grep / FileRead / Glob`

### 6. `BaseAgent.close()` 已纳入 codeintel 生命周期

前一阶段我们已经给 `BaseAgent.close()` 补了 runtime/worktree/llm 收口。

本轮继续补了：

- agent 关闭时自动收口 codeintel manager
- close report 里会出现 `codeintel` 组件

这样 LSP 子进程不会因为 agent 生命周期结束而悬挂在后台。

## 现阶段框架的变换

### 之前

之前 EasyAgent 的 coding 路径主要依赖：

- `Glob`
- `Grep`
- `FileRead`

这意味着 code agent 能做的主要是：

- 文件名定位
- 文本匹配
- 按行读取文件

对于较大仓库，这很快会遇到上限：

- 无法按精确坐标跳定义
- 无法拿到引用列表
- 无法看文件符号树
- 无法看 LSP diagnostics

### 现在

现在框架已经有一层正式的 code intelligence surface：

- 工具层不再只暴露“文件级工具”
- provider 层可以按语言接 LSP
- 结果有统一结构
- unavailable 时有正式 fallback
- 生命周期能被 `agent.close()` 收口

换句话说，EasyAgent 现在已经不只是“能读代码文件”，而是“开始能以符号和 diagnostics 为单位理解代码仓库”。

## 一个具体例子

下面是这轮能力对应的真实工作流：

1. manager 在一个 Python 工作区里注册 `register_codeintel_tools(...)`
2. manager 先调用 `CodeIntelStatus(file_path="sample.py")`
3. 如果返回 `available=true`，再用：
   - `GetDocumentSymbols("sample.py")` 看结构
   - `FindDefinition("sample.py", line=..., column=...)` 跳定义
   - `FindReferences(...)` 看影响面
   - `GetDiagnostics("sample.py")` 看报错/警告
4. 如果返回 `available=false`，则不要继续强行走 LSP，而是直接回退：
   - `Grep`
   - `FileRead`
   - `Glob`

这个流程的意义是：code agent 终于有了一套“优先语义级，再退回文本级”的稳定策略。

## 真实 example

本轮新增的手动调试 example：

- `example/example_phased_codeintel_lsp_v1.py`

它会真实使用：

- `EasyLLM(provider="openai", base_url="http://127.0.0.1:5124/v1", api_key="122", model="qwen3.5-9b")`
- `BasicAgent`
- `register_codeintel_tools(...)`
- `CodeIntelStatus`
- `GetDocumentSymbols`
- `FindDefinition`
- `FindReferences`
- `GetDiagnostics`
- `agent.close()`

注意：

- 它不会被自动执行
- 如果你机器上没有可用的 LSP server，example 依然能演示 fallback 路径

## 验证

本轮新增契约测试：

- `test_lsp_codeintel_provider_round_trip_queries`
- `test_codeintel_tools_return_structured_results`
- `test_codeintel_tools_return_unavailable_fallback_when_server_missing`
- `test_basic_agent_close_reports_codeintel_component`

此外还回归了前一阶段的 close / worktree restore 测试，确保这轮没有破坏 Phase C。

## 结论

按执行计划口径，本轮完成后：

- `Phase D：Code Intelligence v1` 的核心框架能力已落地

下一阶段应进入：

- `Phase E：Hooks / Guardrails + Tool Protocol v2`
