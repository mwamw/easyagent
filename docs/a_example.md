可以把当前框架的 `invoke` 理解成两层：

1. `BasicAgent` 只是入口分发器  
见 [BasicAgent.py](/home/wxd/LLM/EasyAgent/agent/BasicAgent.py:411)

2. 真正干活的是 `invocation_runner` 和 `tool_loop_engine`  
见 [invocation_runner.py](/home/wxd/LLM/EasyAgent/agent/components/invocation_runner.py:90)、[tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:125)

如果只看“最完整、最复杂”的链路，应该看：

`BasicAgent.astream_invoke(...)`  
并且这个 agent 同时启用了：
- `enable_tool=True`
- `context_manager`
- `hook_manager`
- `permission_engine`
- `task_service`
- `agent_runtime / team_manager`
- `execution_context`
- `memory / skills / mailbox`
- provider 走支持真实 usage 的实现

这条链基本覆盖了同步 `invoke` 的全部核心逻辑，只是多了 streaming 和多轮 tool loop。

**总图**

一次最复杂的调用，大致是这条链：

`BasicAgent.astream_invoke`
-> `DefaultInvocationRunner.astream_invoke`
-> 如果 `enable_tool=True`，转到 tool 模式流式执行
-> 组装 system prompt / history / mailbox / memory / skills
-> before-LLM hooks
-> `EasyLLM.astream_with_tools`
-> provider `build_request`
-> provider `async_stream_raw`
-> codec `astream_events`
-> 收到 `tool_calls` 或 `final_response`
-> 如果有工具：执行工具、注入 tool result、提交 pending step、下一轮继续
-> 如果是最终回答：写入 history / trace / observability，结束

**最复杂例子**

假设你有一个 code agent，正在一个 worktree 里修 bug：

- 当前 `execution_context` 带着：
  - `workspaceRoot=/repo`
  - `allowedRoots=[/repo,/tmp/worktrees/bugfix]`
  - `permissionMode=accept_edits`
  - `currentTaskId=task_fix_parser`
  - `worktreePath=/tmp/worktrees/bugfix`
- mailbox 里还有 manager 发来的消息：
  - “先修 parser，再跑 tests，输出 patch summary”
- agent 开了工具、memory、skills、hooks、permissions、runtime/team

然后你调用：

```python
await agent.astream_invoke("修复 parser.py 在空输入时崩溃的问题，并说明原因")
```

这时真实流程是：

1. **入口分发**
`BasicAgent.astream_invoke()` 只是把请求交给 `invocation_runner`。  
见 [BasicAgent.py](/home/wxd/LLM/EasyAgent/agent/BasicAgent.py:496)

2. **判断走普通模式还是工具模式**
`DefaultInvocationRunner` 先看 `agent.enable_tool`。  
如果关了工具，就走单次 LLM 调用。  
如果开了工具，就进入多轮 tool loop。  
见 [invocation_runner.py](/home/wxd/LLM/EasyAgent/agent/components/invocation_runner.py:614)

3. **清理运行时瞬时状态**
开始前会清：
- stop request
- 上次工具中断状态
- 临时 skill 状态

这是为了保证这一轮是干净的。  
见 [invocation_runner.py](/home/wxd/LLM/EasyAgent/agent/components/invocation_runner.py:86)

4. **前置技能改写查询**
`query` 会先过 `skill_manager.on_before_invoke()`。  
如果某个 skill 要补充上下文或改写任务，这一步就发生。  
见 [invocation_runner.py](/home/wxd/LLM/EasyAgent/agent/components/invocation_runner.py:98)

5. **记录 agent run / callback / trace 起点**
会同时启动：
- `callback_manager.on_agent_start`
- trace turn
- observability 的 agent_run

见 [invocation_runner.py](/home/wxd/LLM/EasyAgent/agent/components/invocation_runner.py:284)、[tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:145)

6. **把用户 query 先写入双历史**
框架维护两套历史：
- canonical history
- provider replay history

用户消息先写进这两套历史。  
见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:941)

7. **组装本轮请求输入**
`_build_start_messages()` 会生成 `ReplayRequestInput`。  
它的来源不是只有对话历史，还包括：
- system prompt
- replay history
- context_manager 注入的上下文
- 额外临时 replay
- tools
- reasoning 配置

见 [history_message_assembler.py](/home/wxd/LLM/EasyAgent/agent/components/history_message_assembler.py:60)

8. **system prompt 不是一段字符串，而是一组 block**
`prompt_composer` 会拼这些块：
- identity
- visibility
- task execution
- safety
- tool policy
- tool inventory
- custom instructions
- skill policy / skill listing
- memory
- mailbox

见 [prompt_composer.py](/home/wxd/LLM/EasyAgent/agent/components/prompt_composer.py:79)

9. **mailbox 在这里自动变成运行时输入**
`_build_mailbox_prompt()` 会去 runtime 里读取当前 agent 未消费的 mailbox 消息，并标记为 delivered。  
所以 manager/team/task 发来的消息，会直接进这一轮 system prompt。  
见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:727)

10. **before-LLM hooks 先跑**
正式请求发给模型之前，`hook_manager.before_llm_request()` 可以：
- 改消息
- 改 temperature / reasoning
- 改 kwargs
- 直接阻断请求

见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:524)

11. **EasyLLM 开始走 provider**
这时会进入：
- `_prepare_request_input()`
- `provider.build_request()`
- `provider.async_stream_raw()`
- `codec.astream_events()`

也就是说，真正的 provider 差异是在这里处理，不在 agent 主循环里处理。  
见 [core/llm.py](/home/wxd/LLM/EasyAgent/core/llm.py:521)

12. **streaming 期间，codec 持续吐统一事件**
框架不直接消费 SDK chunk，而是消费 codec 统一后的事件：
- `thinking_delta`
- `text_delta`
- `tool_calls`
- `final_response`

所以上层 tool loop 不需要知道 OpenAI / Responses / Claude / Gemini 的原始流格式。  
例如 OpenAI chat codec 见 [openai_compat/codec.py](/home/wxd/LLM/EasyAgent/core/providers/openai_compat/codec.py:367)

13. **如果模型返回了 tool_calls，不会立刻写正式 history**
这一步很关键。  
框架先把 assistant 的“准备调用工具”这一步放进 `pending_step_state`，而不是马上提交。  
见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:984)

原因是：要把“assistant 发起工具调用”和“tool result”作为一个原子步骤一起提交，避免中途状态不一致。

14. **逐个解析工具调用**
对每个 tool call，会做：
- 取 tool name
- 解析 arguments
- 记 trace 的 tool_call 事件

见 [tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:280)

15. **真正执行工具前，还要先过 before-tool hook**
`_safe_execute_tool_result()` 里先跑：
- `before_tool_use` hook

hook 可以：
- 改参数
- 阻断工具
- 返回结构化错误

见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:3265)

16. **工具执行时，权限系统在这里生效**
真正执行工具时走的是：

```python
tool_registry.execute_tool_result(
    tool_name,
    effective_args,
    permission_context=self.permission_context,
    permission_engine=self.permission_engine,
)
```

所以文件写入、bash、MCP、危险操作，不是工具自己随便跑，而是统一经过 permission engine。  
见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:3322)

17. **工具执行后，还会过 after-tool hook**
工具结果不是立刻回给模型，还会过：
- `after_tool_use` hook

这一步可以：
- 改写 tool result
- 阻断 tool result
- 做 guardrail 清洗

见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:592)

18. **tool result 会变成三种东西**
一个工具结果出来后，会同时变成：
- display string
- canonical history entry
- replay history entry

如果 tool 带 `ephemeral_context`，还会单独保存，准备注入下一轮。  
见 [tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:313)

19. **tool 的临时上下文会注入下一轮**
这一步是 code agent 很重要的点。  
比如代码搜索工具、MCP 资源工具、skill 工具，返回的不一定只是一段文本，还可能是“只对下一轮有效”的临时上下文。  
tool loop 会把这些上下文注入到下一轮请求。  
见 [tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:343)

20. **如果工具需要人工确认，这里会中断**
如果 tool result 状态是 `needs_confirmation`，并且配置要求中断，就会抛 `ToolInterruption`，把 pending step 保留下来。  
见 [tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:321)

21. **一轮 assistant+tool 结束后，才提交 pending step**
`_commit_pending_step_state()` 会把：
- assistant 的 tool-call turn
- tool result turn

一起写入 canonical + replay history。  
见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:1025)

22. **然后可能触发 history compaction**
如果 context_manager 在，且历史超预算，会做 compaction。  
见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:1270)

23. **下一轮重新 build_start_messages**
因为 history、ephemeral context、runtime skill context、mailbox 状态可能都变了，所以不是在原 messages 上硬拼到底，而是可能 rebuild。  
这也是多轮 tool agent 比单轮 chat 更复杂的原因。  
见 [tool_loop_engine.py](/home/wxd/LLM/EasyAgent/agent/components/tool_loop_engine.py:155)

24. **如果模型终于给出 final_response**
这时会：
- 跑 `after_llm_response` hook
- 结束 llm observability
- 把最终 assistant 响应写进 history
- 记录 final trace
- 记录 turn_end
- 结束 agent_run

见 [core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:558)、[core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:2972)

25. **usage 在这一步写入 observability**
现在已经是 provider-first：
- provider 先从真实 response 里取 usage
- agent 在 `_observe_llm_request_end()` 里写入 observability
- stream 场景也不会再把最终 usage 丢掉

见 [core/llm.py](/home/wxd/LLM/EasyAgent/core/llm.py:828)、[core/agent.py](/home/wxd/LLM/EasyAgent/core/agent.py:2972)

**把上面例子串成一句人话**

这次“修 parser”的 `astream_invoke`，实际不是“问一次模型”这么简单，而是：

- 用户问题先进入双历史
- prompt composer 把系统规则、工具说明、memory、mailbox 一起拼好
- hooks 先审一次请求
- provider 发起流式请求
- codec 把原始 chunk 统一成事件
- 模型先返回 `tool_calls`
- agent 执行 `CodeIntel/FileRead/FileEdit/Bash/TaskUpdate/MailboxAck` 之类工具
- 权限系统和 hooks 分别拦截“能不能做”和“内容安不安全”
- tool result 和临时上下文注入下一轮
- 再次请求模型
- 最终返回答案
- 全过程写入 history / trace / runtime / observability

**如果不是最复杂场景，会少哪些步骤**

- `enable_tool=False`：没有 tool loop，直接单轮 LLM 调用
- 没有 `context_manager`：不会做上下文构建和 compaction
- 没有 `hook_manager`：不会有 before/after request/tool 拦截
- 没有 `permission_engine`：工具直接执行，不经过统一权限判定
- 没有 `agent_runtime/team/mailbox`：不会把 mailbox 作为运行时输入
- 不走 stream：不会有 `thinking_delta/text_delta/final_response` 事件循环
