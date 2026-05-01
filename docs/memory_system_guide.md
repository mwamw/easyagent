# Memory System Guide

Memory 模块负责让 Agent 在多轮或跨会话场景下保留信息。  
它和“当前请求上下文”不是一回事：context 解决这次请求带什么，memory 解决长期应该记住什么、如何检索回来。

相关文档：

- [Context And Compaction Guide](./context_and_compaction_guide.md)
- [Session Restore Persistence Guide](./session_restore_persistence_guide.md)
- [Tool System Guide](./tool_system_guide.md)

## 1. Memory 解决什么问题

没有 memory 时，Agent 往往只能依赖：

- 当前对话历史
- 外部数据库
- 手工传入上下文

这会导致：

- 用户偏好不稳定
- 项目长期信息无法复用
- 跨会话恢复体验差

Memory 模块的目标是提供一个统一的长期信息层。

## 2. 核心对象

### `MemoryConfig`

当前主要字段：

- `max_capacity`
- `importance_threshold`
- `decay_factor`
- `max_working_token`
- `batch_size`

这些字段大致分成三类：

- 容量控制
- 重要性/遗忘控制
- 检索与批处理控制

### `MemoryItem`

单条记忆的结构化表示。字段包括：

- `id`
- `content`
- `type`
- `user_id`
- `timestamp`
- `importance`
- `metadata`

### `MemoryType`

当前定义：

- `EPISODIC`
- `WORKING`
- `SEMANTIC`
- `PERCEPTUAL`

### `BaseMemory`

单个 memory store 的抽象接口。它要求实现：

- 添加记忆
- 删除记忆
- 更新记忆
- 搜索记忆
- 清空记忆
- 获取统计信息
- 遗忘策略

### `WorkingMemory`

短周期、高相关度的记忆层，适合当前会话或当前任务。

### `MemoryManage`

这是产品层最常直接使用的对象。  
它负责组合不同类型的 memory store，并统一提供：

- `add_memory`
- `remove_memory`
- `search_memory`
- `update_memory`
- `find_memory`
- `get_memories`
- `forget_memory`
- `get_all_memories`

## 3. 各类 Memory 的定位

### Working Memory

适合：

- 当前任务上下文
- 最近重要事实
- 需要在短期内频繁回看的信息

### Episodic Memory

适合：

- 带时间线的经历
- 发生过的操作 / 事件
- “上次做了什么”

### Semantic Memory

适合：

- 概念
- 规则
- 长期知识

### Perceptual Memory

适合：

- 多模态输入
- 结构化感知结果

## 4. `MemoryManage(...)` 主要参数

### 基础参数

- `config`
- `user_id`

### 开关类参数

- `enable_working`
- `enable_episodic`
- `enable_semantic`
- `enable_perceptual`

### 显式依赖注入

- `working_memory`
- `episodic_memory`
- `semantic_memory`
- `perceptual_memory`

这里的设计意图很明确：

- working/episodic 可以有默认构造路径
- semantic/perceptual 这类更重依赖的 memory，不强行在 `MemoryManage` 初始化时偷偷帮你建

这对框架很重要，因为产品层应该明确控制：

- embedding
- graph/vector store
- 多模态依赖

## 5. 一次 Memory 写入与检索的流程

### 写入

1. 产品或工具调用 `MemoryManage.add_memory(...)`
2. 按 `memory_type` 选择目标 store
3. 构建 `MemoryItem`
4. 写入对应 memory store

### 检索

1. 产品或上下文层传入 query
2. `MemoryManage.search_memory(...)`
3. 在选定 memory types 里搜索
4. 聚合结果
5. 按重要性排序
6. 返回最相关的 `MemoryItem`

## 6. 和 Agent 的集成方式

最常见的接法：

```python
agent = BasicAgent(
    ...,
    memory_manage=memory_manage,
)
```

但这还不够。为了真正让 memory 在推理时生效，通常还需要其中一种：

1. 通过 context manager 把 memory 作为 context source 注入
2. 给 agent 挂 memory builtin tools，让模型主动读写
3. 两者同时做

## 7. Memory Tool 的作用

`register_memory_tools(...)` 会注册 memory 相关 builtin tools。  
这适合让模型显式完成：

- 记录长期信息
- 搜索记忆
- 更新记忆
- 删除过时信息

这类能力很适合：

- 助手型产品
- 长会话 agent
- 个性化 code agent

## 8. Memory 和 Context / Session 的关系

### Memory vs Context

- memory 是长期存储
- context 是本轮请求的输入组织

memory 不会自动等于 prompt；它通常需要：

- 检索
- 过滤
- 格式化
- 再进入 context

### Memory vs Session

- session 保存的是会话快照和消息历史
- memory 保存的是长期知识状态

它们可以一起用，但不应该混成同一层。

## 9. 推荐使用模式

### 最小模式

只接 `WorkingMemory` 和少量 memory tools。  
适合先做产品验证。

### 中等模式

working + episodic，再通过 context manager 检索回注。  
适合长期会话助手。

### 完整模式

working + episodic + semantic + perceptual，并明确接 embedding/vector/graph 依赖。  
适合知识密集型系统。

## 10. 常见坑

### 把 memory 当成 history

这会让记忆层和对话层混在一起，最终两边都不好用。

### 让 semantic/perceptual 在 `MemoryManage` 内部自动隐式初始化

当前设计刻意避免这么做。  
这类依赖应由产品层显式提供。

### 不做检索过滤，直接把所有 memory 塞进 prompt

这样既贵，又破坏 cache，还会稀释真正相关的信息。
