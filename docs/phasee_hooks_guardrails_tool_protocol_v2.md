# Phase E：Hooks / Guardrails + Tool Protocol v2

## 本阶段完成了什么

Phase E 已经落地为当前框架主干能力，不再只是计划项。

本轮完成的核心内容：

- 新增 `core/hooks/`
  - `BaseHook`
  - `HookDecision`
  - `HookManager`
- 新增 `core/guardrails/`
  - `DangerousCommandGuardrail`
  - `SecretLeakGuardrail`
  - `PromptInjectionGuardrail`
  - `build_default_hook_manager()`
- `BaseAgent` 现在正式持有 `hook_manager`
  - 默认自动安装上述 guardrails
  - 支持显式注入自定义 `HookManager`
- hook 生命周期已经接入正式执行链
  - `before_llm_request`
  - `after_llm_response`
  - `before_tool_use`
  - `after_tool_use`
  - `before_compaction`
  - `after_session_restore`
- Tool Protocol v2 已补齐
  - `side_effect_level`
  - `resource_scope`
  - `visibility_scope`
- `ToolRegistry` 已支持同名冲突策略
  - `replace`
  - `error`
  - `keep_existing`
- `ephemeral_context` 已进入
  - trace 事件
  - pending step state
  - session restore 持久化链路
- 高价值工具的提示词和协议元数据已补细
  - `Bash`
  - `FileWrite`
  - `FileEdit`
  - `WebFetch`
  - `CodeIntel*`
  - 多智能体协作相关工具

## 现阶段框架的变换

在 Phase E 之前，EasyAgent 对“能不能执行”只有两层：

- `permission engine`
- `callback`

其中 `permission engine` 负责 allow / deny / ask，`callback` 只能观察，不能阻断或改写。

现在框架多出了一层真正的一等运行时控制面：

- `hook_manager`
- `guardrails`

也就是说，框架现在不只是能判断“某个工具是否允许执行”，还可以在执行前后做更细的内容级控制：

- 改写发给 LLM 的请求
- 改写 LLM 的返回
- 改写工具输入参数
- 改写工具结果
- 阻断危险命令
- 阻断疑似敏感信息外泄
- 标注外部内容中的 prompt injection 风险
- 在 session restore 之后追加框架级恢复处理

与此同时，Tool 协议也不再只有 `read_only / destructive / risk_categories`。  
现在每个工具都可以被更稳定地描述为：

- 它的副作用强度有多高：`side_effect_level`
- 它主要作用于什么资源：`resource_scope`
- 它是 `resident / runtime / turn` 哪种生命周期：`visibility_scope`

这使得 EasyAgent 从“工具集合 + 权限规则”的框架，变成了“带内容级控制、带显式工具协议、可被上层 code agent 产品稳定消费”的框架。

## 一个具体例子

下面这个过程现在是框架内建支持的，而不是上层产品自己硬编码：

1. manager 创建 `BasicAgent`，不额外配置任何 hook。
2. 框架默认安装 guardrails。
3. manager 调用 `Bash(command="rm -rf /")`。
4. `before_tool_use` 先于真正的 shell 执行被触发。
5. `DangerousCommandGuardrail` 命中危险模式，直接阻断。
6. tool result 以 `status=error`、`error_type=guardrail_blocked` 返回给 LLM，而不是先真正执行 shell。

再看另一个例子：

1. manager 调用一个外部内容工具，比如 `WebFetch`，或任何 `resource_scope=["external"]` 的自定义工具。
2. 工具返回的正文里包含 “ignore previous instructions” 之类的注入语句。
3. `after_tool_use` 触发 `PromptInjectionGuardrail`。
4. 工具结果不会被静默原样透传。
5. 返回给模型的 display text 会带 guardrail 警告。
6. 原始 `ephemeral_context` 会被替换成 `guardrail_sanitized_external_context`，避免把不可信外部内容当高优先级上下文继续注入。

这就是 Phase E 的核心意义：  
EasyAgent 现在已经有了真正的“执行前后内容控制层”。

## 本阶段涉及的关键文件

- `core/hooks/`
- `core/guardrails/`
- `core/agent.py`
- `Tool/BaseTool.py`
- `Tool/ToolRegistry.py`
- `agent/components/invocation_runner.py`
- `agent/components/tool_loop_engine.py`
- `agent/components/trace_recorder.py`
- `agent/BasicAgent.py`

## 真实 example

本阶段对应的真实 example：

- `example/example_phasee_hooks_guardrails_tool_protocol_v2.py`

这个 example 使用真实 LLM：

```python
EasyLLM(
    provider="openai",
    base_url="http://127.0.0.1:5124/v1",
    api_key="122",
    model="qwen3.5-9b",
)
```

example 展示的是真实工作流，而不是伪代码：

- 自定义 `HookManager`
- 查看 `ToolSpec v2`
- 危险 shell 被 guardrail 阻断
- 外部内容被 prompt-injection guardrail 标注
- 自定义 `before_llm_request / after_llm_response` hook 如何改写真实 agent 行为
- `after_session_restore` hook 如何给 restore report 追加恢复后处理信息

说明：example 仅用于手动调试，本轮没有执行它。

## 验证

本阶段新增了专门的验证文件：

- `test/test_hooks_and_tool_protocol.py`

本轮确认通过的关键测试包括：

- hooks 能改写 tool 参数
- guardrails 能阻断危险输入
- hooks 能改写 LLM request / response
- Tool Protocol v2 字段能导出
- registry 冲突策略生效
- trace / pending step state 能携带 `ephemeral_context`
- `after_session_restore` hook 生效
- `before_compaction` hook 生效

另外，多组已有关键回归也已通过：

- `test_agent_interfaces.py`
- `test_permissions_and_tasks.py`
- `test_runtime_agents_and_teams.py`
- `test_codeintel.py`
- `test_session_persistence.py` 中与 restore 相关的定向用例

## 下一步

按当前执行计划，Phase E 完成后，下一阶段应进入：

- `Phase F：MCP Engineering`

也就是把 MCP 从“能接上”继续补到“运行时一等扩展面”。
