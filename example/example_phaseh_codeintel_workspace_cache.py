"""Phase H example: codeintel workspace cache + offline snapshot.

This example is intentionally not executed by the agent. It uses the real
EasyLLM configuration requested by the user and demonstrates the full runtime
process introduced in this stage:

1. Register codeintel tools on a real agent/runtime.
2. Prewarm a workspace subtree into the offline cache.
3. Inspect cache status.
4. Save and restore the session.
5. Simulate provider loss and verify offline workspace-symbol fallback.
"""

from __future__ import annotations

import os

from easyagent import BasicAgent, EasyLLM, SessionStore, ToolRegistry
from codeintel import LSPCodeIntelProvider
from core.Config import Config


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DB = os.path.join(PROJECT_ROOT, "db", "phaseh_codeintel_workspace_cache.db")


def build_llm() -> EasyLLM:
    return EasyLLM(
        provider="openai",
        base_url="http://127.0.0.1:5124/v1",
        api_key="122",
        model="qwen3.5-9b",
    )


def main() -> None:
    llm = build_llm()
    registry = ToolRegistry()
    store = SessionStore(SESSION_DB)

    agent = BasicAgent(
        name="phaseh-codeintel-cache",
        llm=llm,
        config=Config(
            workspace_root=PROJECT_ROOT,
            allowed_roots=[PROJECT_ROOT],
        ),
    ).with_tool(registry)

    # Use the default LSP-backed provider first. If the local environment has a
    # matching language server, this will prewarm real symbol/diagnostic data.
    agent.with_codeintel(
        provider=LSPCodeIntelProvider(),
    )

    before = registry.execute_tool_result("CodeIntelCacheStatus", {})
    print("=== Cache Status Before Prewarm ===")
    print(before.to_display_string())

    prewarm = registry.execute_tool_result(
        "CodeIntelPrewarmWorkspace",
        {
            "path_prefix": "codeintel",
            "max_files": 50,
            "include_diagnostics": True,
            "force": False,
        },
    )
    print("=== Prewarm Result ===")
    print(prewarm.to_display_string())

    after = registry.execute_tool_result("CodeIntelCacheStatus", {})
    print("=== Cache Status After Prewarm ===")
    print(after.to_display_string())

    live_symbols = registry.execute_tool_result(
        "GetWorkspaceSymbols",
        {
            "query": "CodeIntelManager",
            "limit": 10,
        },
    )
    print("=== Workspace Symbols (Live Provider Preferred) ===")
    print(live_symbols.to_display_string())

    agent.save_session("phaseh-codeintel-cache", store=store)

    restored = BasicAgent.load_session(
        "phaseh-codeintel-cache",
        llm=build_llm(),
        store=store,
    )

    restored_manager = restored.tool_registry.get_runtime_surface("codeintel_manager", "default")
    print("=== Restore Report ===")
    print(restored.get_last_restore_report())

    restored_cache = restored.tool_registry.execute_tool_result("CodeIntelCacheStatus", {})
    print("=== Restored Cache Status ===")
    print(restored_cache.to_display_string())

    # Simulate provider loss after session restore. This demonstrates the new
    # offline snapshot behavior introduced in this phase.
    healthy_provider = restored_manager.provider
    restored_manager.provider = LSPCodeIntelProvider(server_command=["/definitely/missing/lsp-server"])
    healthy_provider.close()

    offline_symbols = restored.tool_registry.execute_tool_result(
        "GetWorkspaceSymbols",
        {
            "query": "CodeIntelManager",
            "limit": 10,
        },
    )
    print("=== Workspace Symbols (Offline Cache Fallback) ===")
    print(offline_symbols.to_display_string())

    restored.close()


if __name__ == "__main__":
    main()
