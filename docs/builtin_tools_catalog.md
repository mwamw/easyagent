# Builtin Tools Catalog

这份文档是 EasyAgent 内置工具总目录，重点覆盖 `easyagent.tools` 暴露的公共 builtin tools 和注册 helper。目标不是只列名字，而是让产品开发者快速回答这些问题：

- 这个工具是做什么的
- 它是只读还是有副作用
- 在 deferred 模式下是否默认暴露
- 和哪些模块最相关
- 应该通过哪个 helper 批量注册

相关文档：

- [Tool System Guide](./tool_system_guide.md)
- [Tool Authoring Guide](./tool_authoring_guide.md)
- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Permissions Guide](./permissions_guide.md)

## 1. 先理解“工具”和“注册 helper”

EasyAgent 的 builtin tools 有两种使用方式：

### 方式一：直接注册单个工具

例如：

- `register_file_read_tool`
- `register_bash_tool`
- `register_agent_tool`

适合：

- 你只想开放极少数能力

### 方式二：使用 helper 批量注册一组工具

例如：

- `register_filesystem_tools`
- `register_shell_tools`
- `register_task_tools`
- `register_codeintel_tools`

适合：

- 你想快速组装一个某领域能力集

对大多数产品来说，优先用 helper 会更省心。

## 2. Deferred 暴露规则怎么理解

当前正式语义看 `expose_in_deferred`：

- `True`
  - 在 `tool_schema_mode="deferred"` 时默认暴露给模型
- `False`
  - 不默认暴露，需要后续按需展开 schema

对于内置 helper，当前设计是：

- 大多数基础 builtin tools 默认会在 deferred 下暴露
- 但你仍然可以在注册时显式覆盖

## 3. 文件系统工具

最常用的一组只读/编辑工具。

### `FileRead`

- 作用：读取文件内容
- 风险：只读
- 常见场景：
  - 读源码
  - 读配置
  - 读 README / docs
- 常见搭配：
  - `Grep`
  - `Glob`
  - `List`
- 相关模块：
  - Code Agent
  - Context / Prompt 分析型 agent

### `List`

- 作用：列目录结构
- 风险：只读
- 常见场景：
  - 查看项目结构
  - 替代 `ls -al`
- 常见搭配：
  - `FileRead`
  - `Glob`

### `Glob`

- 作用：按模式匹配文件
- 风险：只读
- 常见场景：
  - 搜 `**/*.py`
  - 搜某类配置文件
- 常见搭配：
  - `FileRead`
  - `Grep`

### `Grep`

- 作用：按内容搜索
- 风险：只读
- 常见场景：
  - 搜函数名
  - 搜字符串常量
  - 搜配置项
- 常见搭配：
  - `FileRead`
  - `Glob`

### `FileEdit`

- 作用：对已有文件做精确补丁式编辑
- 风险：写入
- 常见场景：
  - 小范围 patch
  - 精准替换
- 常见搭配：
  - `FileRead`
  - `TaskOutput`

### `FileWrite`

- 作用：写入文件，适合新文件或整段覆盖
- 风险：写入
- 常见场景：
  - 创建新文档
  - 生成新模块
- 常见搭配：
  - `FileRead`

### 推荐 helper：`register_filesystem_tools`

默认会注册：

- `FileRead`
- `List`
- `Glob`
- `Grep`

适合：

- 只读分析型 agent
- 大多数 code agent 的最小只读工具集

## 4. Shell / 后台任务工具

### `Bash`

- 作用：执行 shell 命令
- 风险：高风险 / 有副作用
- 常见场景：
  - 测试
  - 构建
  - git 命令
  - 包管理
- 常见搭配：
  - `TaskOutput`
  - `TaskStop`
- 备注：
  - 应结合 permission engine 使用

### `TaskOutput`

- 作用：读取后台命令输出
- 风险：通常只读
- 常见场景：
  - 看长任务 stdout/stderr

### `TaskStop`

- 作用：停止后台任务
- 风险：有副作用
- 常见场景：
  - 停止卡住的测试
  - 停止长跑构建

### 推荐 helper：`register_shell_tools`

默认会注册：

- `Bash`
- `TaskOutput`
- `TaskStop`

适合：

- 需要跑命令的 code agent
- 带后台任务能力的 runtime

## 5. 基础通用工具

### `Search`

- 作用：网络搜索
- 风险：外部访问
- 常见场景：
  - 查最新资料
  - 查官方文档

### `WebFetch`

- 作用：抓取网页正文
- 风险：外部访问
- 常见场景：
  - 读网页
  - 获取某个 URL 内容

### `Calculator`

- 作用：做简单计算
- 风险：只读
- 常见场景：
  - 数值推导
  - token / 预算辅助计算

### `NotebookEdit`

- 作用：编辑 notebook
- 风险：写入
- 常见场景：
  - 数据科学工作流
  - notebook 修订

### `Config`

- 作用：修改产品配置或运行时配置
- 风险：有副作用
- 常见场景：
  - 切模型
  - 调整配置项

## 6. 交互控制工具

### `AskUserQuestion`

- 作用：向用户请求额外输入或确认
- 风险：交互型
- 常见场景：
  - ask/allow/deny UI
  - 缺参数时补问

### `EnterPlanMode`

- 作用：切到规划模式
- 风险：运行模式切换

### `ExitPlanMode`

- 作用：退出规划模式
- 风险：运行模式切换

这些工具和上层交互产品高度相关，CLI/UI 产品通常都会用到。

## 7. 多 Agent / 协作工具

### `Agent`

- 作用：启动子 agent
- 风险：运行时副作用
- 常见场景：
  - 并行审计
  - 子任务委派

### `AgentGet`

- 作用：读单个子 agent 状态

### `AgentList`

- 作用：列所有子 agent

### `AgentWait`

- 作用：等待子 agent 完成

### `AgentStop`

- 作用：停止子 agent

### `SendMessage`

- 作用：给 agent 或 team 发消息

### `MailboxRead`

- 作用：读取 mailbox 消息

### `MailboxAck`

- 作用：确认 mailbox 消息

### `TeamCreate`

- 作用：创建 team

### `TeamDelete`

- 作用：删除 team

### 推荐 helper

- `register_agent_tool`
  - 只注册 `Agent`
- `register_agent_runtime_tools`
  - 注册 `AgentGet / AgentList / AgentWait / AgentStop`
- `register_mailbox_tools`
  - 注册 `MailboxRead / MailboxAck`

适合：

- Multi-Agent 产品
- Manager/Worker 型架构

## 8. 任务工具

### `TaskCreate`

- 作用：创建结构化任务

### `TaskGet`

- 作用：读取单个任务

### `TaskUpdate`

- 作用：更新任务状态和元数据

### `TaskList`

- 作用：列任务

### `TodoWrite`

- 作用：轻量待办记录
- 与 Task 的区别：
  - Todo 更轻
  - Task 更结构化、更适合持久化协作

### 推荐 helper：`register_task_tools`

适合：

- 带任务树、状态流转、多 agent 协作的产品

## 9. Worktree 工具

### `EnterWorktree`

- 作用：进入新的隔离 worktree
- 风险：运行时副作用

### `ExitWorktree`

- 作用：退出当前 worktree，可选择保留或删除
- 风险：运行时副作用

### 推荐 helper：`register_worktree_tools`

适合：

- code patch worker
- 并行实验
- 子 agent 隔离写入

## 10. CodeIntel 工具

### `CodeIntelStatus`

- 作用：查看 codeintel provider 状态

### `FindDefinition`

- 作用：跳定义

### `FindReferences`

- 作用：查引用

### `GetDocumentSymbols`

- 作用：单文件符号树

### `GetWorkspaceSymbols`

- 作用：工作区符号搜索

### `GetDiagnostics`

- 作用：诊断信息

### `CodeIntelCacheStatus`

- 作用：查看 codeintel 缓存状态

### `CodeIntelPrewarmWorkspace`

- 作用：预热工作区缓存

### 推荐 helper：`register_codeintel_tools`

适合：

- IDE agent
- 仓库导航 agent
- 代码审查 agent

## 11. Memory 工具

Memory 工具通常通过 memory 相关注册逻辑注入，而不是所有产品默认打开。

常见能力包括：

- 写记忆
- 搜索记忆
- 读取记忆
- 更新记忆
- 删除记忆
- 做维护

适合：

- 需要长期记忆的产品
- 用户画像 / 偏好 / 研究型 agent

详见：

- [Memory System Guide](./memory_system_guide.md)

## 12. MCP 工具

MCP 相关 builtin tools 主要分三类：

### 远程 MCP tool 包装器

把 MCP server 暴露的远程 tool 包装成本地 Tool。

### MCP resource tools

- 列资源
- 读资源

### Hub 级资源工具

面向多 server 聚合场景。

### 推荐 helper

- `register_mcp_tools`
- `register_mcp_resource_hub_tools`

详见：

- [MCP Guide](./mcp_guide.md)

## 13. Deferred 模式下最关键的工具

### `ToolSchemaTool`

这是 deferred 模式里的关键基础设施。

- 作用：按需展开工具 schema
- 风险：只读控制工具，不直接对外部系统产生副作用
- 常见场景：
  - 初始只暴露精简目录
  - 需要某个工具时再展开完整 schema

在 `tool_schema_mode="deferred"` 下，它通常应常驻暴露。

详见：

- [Deferred Tools Guide](./deferred_tools_guide.md)

## 14. 按产品形态推荐的 builtin 组合

### 最小只读代码分析 Agent

推荐：

- `register_filesystem_tools`
- `register_codeintel_tools`

### 可执行 code agent

推荐：

- `register_filesystem_tools`
- `register_shell_tools`
- `register_codeintel_tools`
- `register_task_tools`

### Multi-Agent manager

推荐：

- `register_agent_tool`
- `register_agent_runtime_tools`
- `register_mailbox_tools`
- `register_task_tools`

### 企业集成 agent

推荐：

- `register_mcp_tools`
- `register_filesystem_tools`
- `register_task_tools`

## 15. 常见坑

### 坑一：一次性开放过多高风险工具

应该按产品场景最小开放，而不是默认全开。

### 坑二：把 `Bash` 当作唯一工具层

很多场景有更好的 builtin tool，能提供更好的结构化和权限控制。

### 坑三：在 deferred 模式下忘了暴露关键基础工具

例如 `ToolSchemaTool`、基础 filesystem tools。

### 坑四：把任务、协作、worktree 工具混在一个单 agent 产品里全开

这些工具适合有明确 runtime 设计时再打开。
