# Tasks Guide

Task 模块提供 EasyAgent 的结构化任务系统。它和 `TodoWrite` 不是同一层能力：

- `TodoWrite`
  - 更轻量，偏当前对话内的计划和待办记录
- Task
  - 更结构化，可状态流转、可分层、可跨 agent / runtime 使用、可持久化

如果你在做多 Agent 产品、长期任务编排、后台执行队列，Task 模块几乎是必须的。

相关文档：

- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)
- [Permissions Guide](./permissions_guide.md)
- [Callbacks And Streaming Guide](./callbacks_and_streaming_guide.md)

## 1. 核心对象

Task 相关对象主要包括：

- `TaskStatus`
  - 任务状态枚举。
- `TaskRecord`
  - 单个任务记录的数据模型。
- `TaskService`
  - 面向应用的任务操作服务。
- `BaseTaskStore`
  - 抽象存储接口。
- `InMemoryTaskStore`
  - 内存版实现，适合测试和短生命周期运行。
- `SQLiteTaskStore`
  - SQLite 持久化实现，适合单机产品。
- 内置工具：
  - `TaskCreate`
  - `TaskGet`
  - `TaskUpdate`
  - `TaskList`

## 2. `TaskStatus` 有哪些值

当前主要状态有：

- `open`
- `in_progress`
- `blocked`
- `completed`
- `cancelled`

这些状态已经足够覆盖大多数产品工作流。通常不建议一开始就自行扩展出大量自定义状态，先把状态机保持简单会更稳。

## 3. `TaskRecord` 字段逐项解释

`TaskRecord` 是结构化任务的核心数据模型。主要字段有：

- `task_id`
  - 任务唯一 ID。默认由 `TaskService` 生成，也可显式指定。
- `title`
  - 任务标题。
- `description`
  - 详细描述。
- `status`
  - 当前任务状态。
- `owner`
  - 所属 agent、用户或工作流节点。
- `parent_task_id`
  - 父任务 ID，用于构建层级任务树。
- `metadata`
  - 自定义附加字段。适合存：
    - ticket id
    - repository
    - priority
    - workflow stage
    - routing hints
- `created_at`
  - 创建时间。
- `updated_at`
  - 最近更新时间。

一个很重要的设计点是：Task 模块把“结构化业务字段”和“灵活扩展字段”分开了。

- 稳定通用字段进顶层
- 产品自定义字段进 `metadata`

## 4. `TaskService` 负责什么

`TaskService` 是推荐给上层应用直接使用的入口。它主要负责：

1. 创建任务
2. 获取任务
3. 更新任务
4. 列表查询任务

它本身不关心任务存在哪，只依赖 `BaseTaskStore`。

所以一般产品代码只需要决定两件事：

- 用什么 store
- 怎么把 `TaskService` 接到 Agent

## 5. `TaskService` 的主要方法

### `create_task(...)`

主要参数：

- `title`
- `description`
- `status`
- `owner`
- `parent_task_id`
- `metadata`
- `task_id`

默认会生成：

- `task_id`
- `created_at`
- `updated_at`

### `get_task(task_id)`

根据 `task_id` 读取任务。若不存在，会抛 `TaskNotFoundError`。

### `update_task(...)`

支持更新：

- `title`
- `description`
- `status`
- `owner`
- `parent_task_id`
- `metadata`

其中 `metadata` 支持：

- `merge_metadata=True`
  - 在旧 metadata 基础上 merge
- `merge_metadata=False`
  - 全量替换

### `list_tasks(...)`

支持按以下维度查询：

- `status`
- `owner`
- `parent_task_id`
- `metadata_filters`
- `limit`

其中 `metadata_filters` 支持基于 metadata 做过滤，很适合产品按项目、队列、租户做筛选。

## 6. `InMemoryTaskStore` 和 `SQLiteTaskStore` 的区别

### `InMemoryTaskStore`

优点：

- 零依赖
- 启动快
- 适合测试

缺点：

- 进程退出后数据消失

适合：

- 单测
- demo
- 短生命周期 agent runtime

### `SQLiteTaskStore`

优点：

- 单文件持久化
- 部署简单
- 足够支撑单机产品和小规模服务

缺点：

- 不适合高并发分布式场景

适合：

- CLI / desktop / local server
- 单机控制面的多 agent 产品

## 7. 一次典型执行流程

下面是一条常见链路：

1. 用户要求“把仓库 cache 重构拆成若干任务”
2. 主 Agent 调用 `TaskCreate` 创建：
   - 总任务
   - 子任务 A/B/C
3. 子 agent 各自领取任务，运行中通过 `TaskUpdate` 更新状态：
   - `open -> in_progress`
   - `in_progress -> blocked`
   - `in_progress -> completed`
4. 主 Agent 用 `TaskList` 聚合所有子任务状态
5. 最终向用户汇总整个任务树进度

这一套比只靠 `TodoWrite` 强很多，因为它有：

- ID
- owner
- 父子关系
- 可持久化状态

## 8. 和 `BasicAgent` 的集成

最常见的集成方式：

```python
from easyagent import BasicAgent, EasyLLM
from easyagent.tasks import TaskService, SQLiteTaskStore
from easyagent.tools import register_task_tools
from Tool.ToolRegistry import ToolRegistry

store = SQLiteTaskStore("db/tasks.db")
task_service = TaskService(store)

registry = ToolRegistry()
register_task_tools(registry, task_service=task_service)

agent = BasicAgent(
    name="manager",
    llm=EasyLLM(),
    enable_tool=True,
    tool_registry=registry,
    task_service=task_service,
)
```

这里有两个关键点：

1. `task_service` 给 agent/runtime 直接使用
2. task builtin tools 让模型能结构化操作任务

## 9. Task 和 Runtime / Team / Agent 的关系

Task 模块和多 Agent 运行时关系很强。

常见搭配：

- `Agent`
  - 让子 agent 负责某个 task
- `SendMessage`
  - 推送任务状态变化
- `TeamCreate`
  - 用 team 管理某组任务的执行者

推荐理解：

- runtime 决定谁去做
- task 决定做什么、做到哪一步了

## 10. Task 适合解决哪些产品问题

很适合：

- 长任务拆分
- 多 agent 协作
- 后台工作流状态跟踪
- 人工介入和恢复
- 带 owner 的任务路由

不适合：

- 只想让模型记一下当前回复里的 3 个小待办

后者更适合 `TodoWrite`。

## 11. 推荐的任务建模方式

### 模式一：父子树

- 父任务：大目标
- 子任务：具体执行项

适合：

- 代码重构
- 多模块审计
- 发布流程

### 模式二：owner 驱动

- 每个任务有 `owner`
- Agent 或用户按 owner 过滤任务

适合：

- 多 worker 协作

### 模式三：metadata 驱动

把路由和业务标签放进 metadata：

- `repo`
- `priority`
- `tenant`
- `workflow_stage`

适合：

- 产品内做灵活筛选，而不污染顶层模型

## 12. 常见坑

### 坑一：把 Task 当成 Todo 的替代品

Task 更重。如果只是当前对话内临时计划，用 `TodoWrite` 就够了。

### 坑二：所有业务字段都塞顶层

产品自定义维度优先放 `metadata`，不要轻易改 Task 核心模型。

### 坑三：没有 owner 和 parent_task_id

这样任务会很快失去可追踪性。

### 坑四：把任务状态写死在 prompt 里

状态应该来自 `TaskService` 和 store，而不是只靠模型口头描述。

### 坑五：没有持久化就做长任务

如果任务会跨轮、跨 agent、跨进程，至少用 `SQLiteTaskStore`。
