# Worktree Guide

Worktree 模块提供隔离工作区能力。它对 Code Agent 尤其重要，因为很多实际产品场景都要求：

- 子 agent 可以在独立副本里修改代码
- 主工作树不能被临时实验污染
- 多个并行方案互不干扰

这正是 Git worktree 适合解决的问题。

相关文档：

- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)
- [CodeIntel Guide](./codeintel_guide.md)
- [Builtin Tools Catalog](./builtin_tools_catalog.md)

## 1. 核心对象

这一层最重要的对象有：

- `WorktreeManager`
  - 运行时管理器，负责创建、进入、退出、清理 worktree。
- `GitWorktreeInfo`
  - 单个 worktree 的静态信息。
- `GitWorktreeSession`
  - 当前活动 worktree 会话。
- 内置工具：
  - `EnterWorktree`
  - `ExitWorktree`

## 2. `GitWorktreeInfo` 记录什么

它描述一个 worktree 的基本状态：

- `path`
  - worktree 路径
- `branch`
  - 绑定的分支名
- `head`
  - 当前 HEAD 提交
- `bare`
  - 是否 bare
- `detached`
  - 是否 detached HEAD

它适合做：

- UI 展示
- 会话恢复
- runtime 状态同步

## 3. `GitWorktreeSession` 记录什么

这是“当前活动 worktree 会话”的状态对象。主要字段有：

- `original_cwd`
  - 进入 worktree 前的原始工作目录
- `worktree`
  - 当前 worktree 的 `GitWorktreeInfo`
- `base_head`
  - 创建时的基线提交
- `created_at`
  - 创建时间

之所以要单独有 session，而不只记录一个 path，是因为退出时往往需要知道：

- 原始 cwd
- 是否有未提交变更
- 相对基线前进了多少提交

## 4. `WorktreeManager` 做什么

这是 worktree 模块的真正总控。

它主要负责：

1. 检测仓库根目录
2. 创建 worktree
3. 进入 worktree，建立活动 session
4. 退出 worktree
5. 按需移除 worktree
6. 清理 `git worktree prune`
7. 跟踪受自己管理的 worktree

推荐理解：

- `GitWorktreeInfo` 是静态信息
- `GitWorktreeSession` 是一次使用中的会话
- `WorktreeManager` 是生命周期控制器

## 5. `WorktreeManager` 初始化参数

最常用参数有：

- `repo_root`
  - Git 仓库根目录。
- `git_binary`
  - 使用哪个 git 可执行文件。
- `storage_dir`
  - worktree 目录的存放位置。默认放在仓库旁边的 `.easyagent-worktrees`。
- `original_cwd`
  - 进入 worktree 前的工作目录。

如果你在做产品，通常建议显式指定：

- `repo_root`
- `storage_dir`

这样更可控。

## 6. `WorktreeManager` 的主要方法

### `detect_repo_root(start_path)`

从任意路径向上解析 Git 根目录。

### `create_worktree(name, base_ref="HEAD", branch_prefix="easyagent/")`

创建一个新的 Git worktree，并生成独立分支。

### `list_worktrees()`

列出仓库当前所有 worktree。

### `enter_worktree(...)`

创建并切入一个新的活动 worktree session。

### `exit_worktree(action, discard_changes=False)`

退出当前活动 session。

`action` 只支持：

- `keep`
- `remove`

### `remove_worktree(path, force=False)`

移除指定 worktree。

### `prune()`

执行 `git worktree prune`。

## 7. 一次典型执行流程

一个典型的 code agent 子任务流程是：

1. 主 agent 决定把某个实现任务交给子 agent
2. runtime 使用 `WorktreeManager.enter_worktree(...)`
3. 创建新的 worktree 和分支
4. 子 agent 在该路径中运行：
   - 读文件
   - 修改文件
   - 跑测试
5. 任务结束时：
   - 如果需要保留成果：`ExitWorktree(action="keep")`
   - 如果是临时实验且不保留：`ExitWorktree(action="remove")`

这条链路的关键收益是：主工作树不会被并行任务弄脏。

## 8. 如何接入 `BasicAgent`

通常 worktree 不会单独成为 `BasicAgent` 的核心参数，而是作为 runtime / subagent builder 的一部分接入。

常见方式：

```python
from Tool.runtime.worktree_manager import WorktreeManager
from easyagent.tools import register_worktree_tools
from Tool.ToolRegistry import ToolRegistry

manager = WorktreeManager(repo_root=".")
registry = ToolRegistry()
register_worktree_tools(registry, worktree_manager=manager)
```

然后：

- 主 agent 可以调用 `EnterWorktree`
- 子 agent 在新的 worktree 中执行

## 9. 和 Runtime / CodeIntel 的关系

### 和 Runtime 的关系

worktree 经常作为多 agent runtime 的隔离模式使用。

它解决的是：

- 并行任务互不影响
- 子任务有独立写空间

### 和 CodeIntel 的关系

worktree 一旦切换，`workspace_root` 也通常应该切换。

否则会导致：

- codeintel 跳转结果不准
- workspace cache 混乱

因此在做 code agent 产品时，worktree 和 codeintel 应该联动设计。

## 10. `EnterWorktree` / `ExitWorktree` 两个内置工具怎么用

### `EnterWorktree`

作用：

- 创建新的 worktree
- 建立当前活动 session

适合：

- 子 agent 启动前
- 需要隔离写入前

### `ExitWorktree`

作用：

- 退出当前活动 session
- 决定保留还是移除 worktree

适合：

- 子任务结束
- 中止实验性改动

## 11. 推荐的产品使用模式

### 模式一：后台 worker 默认 worktree 隔离

适合：

- code patch worker
- multi-agent 修复系统

### 模式二：只有写任务才进入 worktree

读任务仍在主工作树执行。

适合：

- 成本敏感的系统

### 模式三：实验型分支隔离

多个方案各自进不同 worktree，对比后再决定保留哪个。

适合：

- 自动重构
- 多方案 patch 生成

## 12. 常见坑

### 坑一：进入 worktree 后仍然用主工作树路径

这会导致：

- 写到错误目录
- codeintel/workspace cache 错位

### 坑二：所有子任务共用一个活动 session

worktree 隔离的意义会大打折扣。

### 坑三：退出时一律 remove

有些任务需要保留 worktree 供人工检查或后续提交。

### 坑四：忘记清理临时 worktree

长时间运行的系统需要定期 prune 和清理。

### 坑五：把 worktree 当成权限系统

worktree 只是隔离工作目录，不替代 permission、confirmation 和 hook。
