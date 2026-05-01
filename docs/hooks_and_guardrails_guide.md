# Hooks And Guardrails Guide

Hook 系统负责在关键执行点**阻断、修改或放行** payload。  
如果 callback 是旁观层，hook 就是干预层。

相关文档：

- [Callbacks And Streaming Guide](./callbacks_and_streaming_guide.md)
- [Permissions Guide](./permissions_guide.md)

## 1. Hook 解决什么问题

有些规则不只是“记录一下”，而是必须真正改变行为，例如：

- 某类 prompt 禁止发给模型
- 某些工具参数需要重写
- 某些 response 必须被审查

这类需求如果塞进 callback 会很别扭，塞进 tool 本身又会把策略逻辑写死。  
Hook 就是解决这个问题的。

## 2. 核心对象

### `HookAction`

当前动作：

- `allow`
- `modify`
- `block`

### `HookDecision`

单个 hook 返回的决定。关键字段：

- `action`
- `message`
- `updates`
- `metadata`
- `error_type`

### `HookExecutionResult`

整个 hook stage 的聚合结果。关键字段：

- `payload`
- `audit`
- `blocked`
- `message`
- `error_type`

### `BaseHook`

可选地覆盖框架定义的各个 hook 点。

### `HookManager`

负责按阶段顺序执行 hooks，并聚合出最终结果。

## 3. 主要 hook 点

### `before_llm_request`

在请求发给模型前执行。  
适合：

- 拦敏感信息
- 加审计标签
- 阻止某类 prompt

### `after_llm_response`

在模型返回后执行。  
适合：

- 结果审查
- 响应规范化
- 思考内容清洗

### `before_tool_use`

在工具执行前执行。  
适合：

- 二次校验参数
- 拦截高风险命令
- 给 payload 加补充上下文

### `after_tool_use`

在工具执行后执行。  
适合：

- 结果清洗
- 错误增强
- 审计记录

### `before_compaction`

在 history compaction 前执行。  
适合：

- 阻止某些历史被压缩
- 打 compaction 审计点

### `after_session_restore`

在 session 恢复后执行。  
适合：

- 二次检查缺失依赖
- 恢复后补写状态

## 4. 一个最小自定义 hook

```python
from easyagent.hooks import BaseHook, HookDecision

class BlockDangerousShellHook(BaseHook):
    def before_tool_use(self, payload):
        if payload.get("tool_name") == "Bash":
            command = str(payload.get("parameters", {}).get("command", ""))
            if "rm -rf" in command:
                return HookDecision.block("检测到危险命令，已阻止执行。")
        return None
```

## 5. 和 Agent 的集成

```python
from easyagent.hooks import HookManager

hook_manager = HookManager([BlockDangerousShellHook()])
agent = BasicAgent(..., hook_manager=hook_manager)
```

## 6. 和 Permission / Callback 的区别

### Hook vs Permission

- permission 负责“能不能做”
- hook 负责“要不要改、要不要拦、要不要附加审计”

### Hook vs Callback

- callback 主要观测
- hook 主要干预

## 7. 推荐使用场景

适合写 hook 的规则：

- prompt injection 防护
- secret / token 泄露扫描
- 危险 shell 命令阻断
- session 恢复后校验

## 8. 常见坑

### 把权限规则写成 hook

会让规则体系分散。  
通用的 allow / ask / deny 应优先走 permission engine。

### hook 里做过重逻辑

hook 在关键路径上，太重会直接拖慢主链路。
