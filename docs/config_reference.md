# Config Reference

`Config` 是 EasyAgent 的全局运行配置对象。  
它不直接拥有所有运行状态，但决定了 Agent、Tool、Cache、Runtime、Session、Worktree 等子系统的默认行为。

相关文档：

- [README](../README.md)
- [Agent Guide](./agent_guide.md)
- [Tool System Guide](./tool_system_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 1. 最小示例

```python
from easyagent import Config

config = Config(
    tool_schema_mode="deferred",
    workspace_root=".",
    allowed_roots=["."],
    enable_worktree=False,
)
```

## 2. 字段总览

### LLM 默认值

- `default_model`
  - 默认模型名
- `default_provider`
  - 默认 provider
- `temperature`
  - 默认温度
- `max_tokens`
  - 默认最大输出 token

这些字段主要在你没有显式给 `EasyLLM(...)` 传参时作为默认值参考。

### 调试与日志

- `debug`
  - 是否启用调试模式
- `log_level`
  - 日志级别

### 历史与压缩

- `max_history_length`
  - 触发上下文压缩之前允许保存的历史长度
- `trigger_threshold`
  - 接近阈值时开始考虑 compaction 的保守触发点
- `persist_reasoning_history`
  - 是否持久化 reasoning / thinking 历史

这组字段与：

- [Context And Compaction Guide](./context_and_compaction_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

关系最大。

### 工作区与命令执行

- `workspace_root`
  - 工具和 code agent 默认工作根目录
- `allowed_roots`
  - 文件系统工具允许访问的路径列表
- `shell`
  - shell 工具默认 shell
- `command_timeout_ms`
  - shell / process 工具默认超时
- `max_background_tasks`
  - 后台任务上限
- `git_binary`
  - git 二进制路径
- `enable_worktree`
  - 是否默认启用 worktree 相关能力

这组字段主要作用于：

- builtin filesystem tools
- shell / background task
- subagent / worktree runtime

### 工具与权限交互

- `interrupt_on_confirmation`
  - 工具需要用户确认时，是否中断当前 invoke 让上层先处理确认

如果你的上层产品有“Ask -> 用户确认 -> Allow”流程，这个字段很重要。

### Cache 相关

- `cache_policy`
  - `PromptCachePolicy` 对象，定义 cache 是否启用、模式、TTL、scope、breakpoint strategy
- `tool_schema_mode`
  - `full` 或 `deferred`
- `stable_tool_order`
  - 是否按稳定顺序导出工具
- `record_cache_breaks`
  - 是否记录 cache break 诊断
- `subagent_inherit_cache_safe_params`
  - 子 agent 是否继承父级 cache-safe 参数
- `subagent_cache_policy`
  - `inherit | read_only | skip_write | isolated`
- `google_cached_content_name`
  - Google cached content 名称，供支持该能力的后端复用

重点阅读：

- [Deferred Tools Guide](./deferred_tools_guide.md)
- [Observability And Cache Guide](./observability_and_cache_guide.md)

## 3. `PromptCachePolicy`

`cache_policy` 的核心字段：

- `enabled`
  - 是否启用 cache 规划
- `mode`
  - `auto`：默认策略
  - `read_only`：优先复用，不主动写新 cache
  - `write`：显式允许写新 cache
  - `skip_write`：不要因为这次请求更新新的 cache marker
- `ttl`
  - provider 支持时使用的 TTL
- `scope`
  - `session` 或其他 provider 适配层定义的范围
- `breakpoint_strategy`
  - 请求编排时选择怎样的 cache breakpoint 规划

## 4. 环境变量

`Config.from_env()` 当前支持的重点环境变量包括：

- `DEBUG`
- `LOG_LEVEL`
- `TEMPERATURE`
- `MAX_TOKENS`
- `WORKSPACE_ROOT`
- `ALLOWED_ROOTS`
- `SHELL`
- `COMMAND_TIMEOUT_MS`
- `MAX_BACKGROUND_TASKS`
- `GIT_BINARY`
- `ENABLE_WORKTREE`
- `INTERRUPT_ON_CONFIRMATION`
- `PERSIST_REASONING_HISTORY`
- `PROMPT_CACHE_ENABLED`
- `PROMPT_CACHE_MODE`
- `PROMPT_CACHE_TTL`
- `PROMPT_CACHE_SCOPE`
- `PROMPT_CACHE_BREAKPOINT_STRATEGY`
- `CACHE_DYNAMIC_MEMORY`
- `CACHE_DYNAMIC_MAILBOX`
- `CACHE_TURN_SKILLS`
- `TOOL_SCHEMA_MODE`
- `STABLE_TOOL_ORDER`
- `RECORD_CACHE_BREAKS`
- `SUBAGENT_INHERIT_CACHE_SAFE_PARAMS`
- `SUBAGENT_CACHE_POLICY`
- `GOOGLE_CACHED_CONTENT_NAME`

## 5. 常见配置模板

### 最小聊天 Agent

```python
Config()
```

### 工具型 Code Agent

```python
Config(
    tool_schema_mode="deferred",
    workspace_root=".",
    allowed_roots=["."],
    cache_policy={"enabled": True, "mode": "auto"},
)
```

### 多 Agent 协作

```python
Config(
    tool_schema_mode="deferred",
    max_background_tasks=8,
    subagent_cache_policy="read_only",
)
```

### 偏保守的高风险环境

```python
Config(
    interrupt_on_confirmation=True,
    enable_worktree=True,
    tool_schema_mode="deferred",
)
```

## 6. 和 Agent 的集成方式

```python
agent = BasicAgent(
    name="assistant",
    llm=llm,
    config=config,
)
```

通常 `Config` 会同时影响：

- `BasicAgent` 的默认运行行为
- builtin tool helper 的默认参数来源
- request compiler 的 cache / reminder / prompt 规划
- runtime 的 subagent 策略

## 7. 常见误区

### `Config` 会替代 `EasyLLM(...)` 吗

不会。  
`EasyLLM` 负责 provider 和模型连接；`Config` 负责框架运行行为。

### `tool_schema_mode="deferred"` 会让所有工具永远不可见吗

不会。  
它只是不全量暴露 schema。详见：

- [Deferred Tools Guide](./deferred_tools_guide.md)

### Placement 会影响缓存分区吗

不会。`PromptBlock.placement` 决定 system/system-reminder 消息位置；`cache_partition` 和 `cacheable` metadata 决定缓存策略。Memory retrieval 和 mailbox 由动态上下文管线管理，不通过 Config 开关移动到系统提示词。
