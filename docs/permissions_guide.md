# Permissions Guide

Permission 系统负责回答一个关键问题：  
**模型决定调用某个工具时，这个工具到底是立即执行、要求确认，还是应该被拒绝？**

这不是 UI 功能，而是框架的核心安全边界之一。

相关文档：

- [Tool System Guide](./tool_system_guide.md)
- [Hooks And Guardrails Guide](./hooks_and_guardrails_guide.md)
- [Runtime Collaboration Guide](./runtime_collaboration_guide.md)

## 1. 核心对象

### `PermissionBehavior`

三种结果：

- `allow`
- `deny`
- `ask`

### `PermissionMode`

当前模式包括：

- `default`
- `plan`
- `accept_edits`
- `dont_ask`
- `bypass`

### `RiskCategory`

当前风险类别包括：

- `filesystem_read`
- `filesystem_write`
- `shell`
- `network`
- `process`
- `mcp`
- `side_effect`

### `PermissionRule`

描述一条规则。字段至少包括：

- `tool_name`
- `behavior`
- `matcher`
- `source`
- `description`

### `PermissionDecision`

表示最终判定结果。字段包括：

- `behavior`
- `tool_name`
- `reason`
- `matched_rule_source`
- `risk_categories`
- `requires_confirmation`
- `metadata`

### `PermissionStore`

规则存储层，支持不同 source 和优先级。

### `PermissionContext`

当前会话或当前 invoke 的权限上下文。它持有：

- `mode`
- `rules`
- `store`
- `metadata`

### `PermissionEngine`

真正负责执行判定逻辑。

## 2. 一次权限判定的流程

典型流程如下：

1. 模型返回某个 tool call
2. `ToolRegistry` 先拿到 `ToolSpec`
3. `PermissionEngine.authorize(...)`
4. 根据 tool、parameters、context 推导风险类别
5. 查找命中的规则
6. 结合当前 mode 得出最终结论
7. 返回 `PermissionDecision`

如果结果是：

- `allow`
  - 工具继续执行
- `ask`
  - Agent 中断，等待上层应用确认
- `deny`
  - 返回拒绝结果，不执行工具

## 3. `PermissionMode` 的实际意义

### `default`

普通模式，规则优先，剩余情况按风险判断。

### `plan`

规划模式。  
高风险写操作、网络、副作用类动作应默认被挡住。

### `accept_edits`

偏向允许文件编辑类操作，但不等于所有高风险操作都放开。

### `dont_ask`

不允许弹确认。  
因此凡是必须确认的能力，通常会直接拒绝。

### `bypass`

绕过权限系统，基本等价全放行。  
只适合你非常确定的内部场景。

## 4. `requires_confirmation` 与 Permission 的关系

很多人容易混淆：

- `Tool(..., requires_confirmation=True)`
- permission rule 里要求 `ask`

两者关系是：

- tool 自己声明它天生危险
- permission system 根据更大的上下文决定最终如何处理

也就是说，工具字段是语义输入，PermissionEngine 才是最终裁判。

## 5. `matcher` 应怎么理解

`PermissionRule.matcher` 用来做更细粒度匹配，例如：

- 路径范围
- 参数模式
- 命令片段
- 风险标签

框架层的目标是统一判定入口，不是把所有产品策略都硬编码在 tool 里。

## 6. 和 Agent 的集成方式

最常见接法：

```python
agent = BasicAgent(...).with_permissions(
    engine=PermissionEngine(),
    context=PermissionContext(),
)
```

然后由上层在不同会话里切换：

- mode
- rules
- source

## 7. 上层应用如何实现 Ask / Allow / Deny

EasyAgent 的定位不是替你做确认 UI，而是统一生成确认语义。

典型产品流：

1. agent 调某个工具
2. 权限结果为 `ask`
3. tool loop 中断
4. 上层 UI 展示：
   - 工具名
   - 参数
   - 风险类别
   - reason
5. 用户选择 allow / deny
6. 若 allow，上层再恢复执行

所以“用户确认执行”通常是上层应用实现，但其协议和中断点由 EasyAgent 提供。

## 8. 和 Runtime / Team 的关系

Permission 不应该只对主 agent 生效。  
在多 Agent 系统里，子 agent 也必须继承或覆盖权限上下文。

否则会出现：

- 主 agent 很谨慎
- 子 agent 却默认全放开

这会直接破坏整个系统的安全边界。

## 9. 推荐实践

### 本地 Code Agent

- `default` 或 `accept_edits`
- shell / network / process 默认 ask 或 deny

### 纯计划模式

- `plan`
- 强制禁止高风险副作用工具

### 受控自动化

- 只对白名单工具 allow
- 其余 ask / deny

## 10. 常见坑

### 把确认逻辑写死在 UI 层

这样框架层无法统一表达“当前为什么中断”。

### 只看工具名，不看参数

很多风险其实来自参数，而不是工具名本身。

### `dont_ask` 理解成“全自动执行”

恰恰相反，它通常意味着“不允许进入确认流程”，因此高风险动作应直接拒绝。
