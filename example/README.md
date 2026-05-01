# Example Index

当前示例按两类理解：

## Framework API Examples

这类示例主要演示框架公共 API、运行时能力和子系统接入。

- `example_phase1a_permission_task_session.py`
- `example_phase23_runtime_collaboration_restore.py`
- `example_phase23_runtime_collaboration_final.py`
- `example_phase23_mailbox_collaboration_complete.py`
- `example_phasec_restore_report_lifecycle.py`
- `example_phased_codeintel_lsp_v1.py`
- `example_phaseh_codeintel_workspace_cache.py`
- `example_phasei_observability_metrics.py`
- `example_phasej_provider_usage_extraction.py`
- `example_phasee_hooks_guardrails_tool_protocol_v2.py`
- `example_phasef_mcp_engineering.py`

## Product-Like Examples

这类示例更像“基于框架做出的具体助手/工作流”。

- `example_code_agent_product_bootstrap.py`
- `example_coding_workflow.py`
- `example_research_and_memory.py`
- `example_mcp_filesystem_updated.py`
- `example_builtin_tools.py`
- `example_skill_runtime.py`

## Streaming / Provider / Misc

- `example_stream.py`
- `example_stream_display.py`
- `example_stream_history.py`
- `example_openai_compat_cache_probe.py`
- `openai_res.py`
- `agent_test.py`

## 当前推荐阅读顺序

如果你是第一次接触当前框架，建议按这个顺序看：

1. `example_phaseg_sdk_release.py`
2. `example_code_agent_product_bootstrap.py`
3. `example_phase1a_permission_task_session.py`
4. `example_phase23_mailbox_collaboration_complete.py`
5. `example_phasec_restore_report_lifecycle.py`
6. `example_research_and_memory.py`
7. `example_phased_codeintel_lsp_v1.py`
8. `example_phaseh_codeintel_workspace_cache.py`
9. `example_phasei_observability_metrics.py`
10. `example_phasej_provider_usage_extraction.py`
11. `example_phasef_mcp_engineering.py`

## 说明

- 这些 example 不保证都会自动执行
- 阶段性 example 更适合阅读“框架能力是怎么演进的”
- 如果你要从公共 SDK 边界开始接入，优先使用 `from easyagent import ...`
