# Context 与压缩

Context 接受已经准备好的内容，不查询 Memory、RAG、文件系统或会话数据库。
旧的 add_source、ContextSource、来源权重和来源配额已删除。

## 配置与组装

```python
from easyagent.context import ContextManager, ContextItem, LLMHistoryCompactor

context = ContextManager(max_tokens=24_000)
context.set_history_compactor(LLMHistoryCompactor(llm=llm, language="en", recent_turns=4))
agent.with_context(context)

request = context.build_request_input(
    query="Explain this document",
    system_prompt="You are a helpful assistant.",
    replay_history=[],
    provider_name=llm.provider_name,
    context_items=[ContextItem(content=document_text, source="document")],
)
context.validate_request(request)
```

build_request_input 构造候选请求，validate_request 检查最终完整输入。
Agent 的请求管线自动完成组装、压缩和最终检查；直接使用 Context 时调用方负责检查。
未配置 max_tokens 表示不设置框架输入上限，不代表模型具有无限上下文。
来源标签仅用于追踪和展示，不参与配额分配。

## Token 计数优先级

Agent 优先使用最近一次主模型请求返回的真实 usage 作为基线，再用本地计数器
估算基线之后的变化：`当前占用 ≈ API 基线 + 当前本地计数 - 基线本地计数`。
有效 API 基线优先，不取 API 与本地估算的较大值。新增用户消息、工具结果和
临时提醒的移除均计入增量；移除内容允许产生负增量。

响应完整保留时，基线包含 API 输入和输出；隐藏推理或被 Hook 改写的响应不
直接复用原输出计数，而是使用 API 输入基线加实际保留内容的本地增量。
Anthropic 的缓存读取/写入 token 计入上下文，OpenAI 已包含在输入里的缓存
token 不重复相加；这些计数不等同于费用或累计调用量。

首轮、API 未提供有效 usage，或模型、固定提示词、工具配置等发生变化时，
退回完整本地估算。清空、恢复或压缩历史也会使旧基线失效；之后的成功响应
重新建立基线。每个 Agent 独立维护基线，不依赖 Observability，也不被共用
LLM 的摘要请求或其他 Agent 覆盖。同步、异步、流式请求使用同一规则。

自动压缩预检、发送前最终预算检查、`agent.get_context_usage()` 使用同一优先级。
返回的 `estimatedRequestTokens` 是优选值，`localEstimatedRequestTokens` 仅供诊断；
`request_estimate_source` 为 `provider_usage_plus_delta_estimate` 或
`local_request_estimate`。新请求未发送前，其增量仍是估算，不是 API 实测。
直接使用未接入 Agent usage 跟踪的 Context 时，仅做本地估算。

## 动态信息与扩展

模型主动检索使用普通 Tool/ToolResult。需要自动检索时，由模块在
before_context/abefore_context Hook 中自行查询，再调用 MetaMessage。
Context 不持有模块实例。异步请求会等待异步 Hook 和 acompact_persistent_history。

使用 BaseContextManager 替换整个上下文实现，或仅替换 BaseHistoryCompactor。
压缩器生成结果，执行器在最终预算检查成功后提交历史更新。
临时 MetaMessage 不进入旧历史摘要，仍由自身生命周期回收。
不能压缩到预算以内时抛出 ContextBudgetExceeded；禁止发送超限请求。

## History 与持久化

ConversationHistory 是现有的内存组件，维护 canonical 和 provider replay 两种表示。
本次重构没有新增完整 transcript journal；持久化保存当前可恢复的历史状态。
压缩可能替换当前历史，因此不要把当前快照宣称为完整原始会话归档。
查看 token 用量不运行检索 Hook，也不会发起模型请求。
