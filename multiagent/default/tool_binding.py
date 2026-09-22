"""Tool selection and binding policy for the built-in multi-agent harness.

Tool itself has no child lifecycle. Ordinary tools are reusable capabilities;
agent-bound tools are rebuilt here or supplied by the caller's tool factories.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable, Mapping

from runtime import AgentRuntime
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.bash_tool import BashTool, register_shell_tools
from Tool.builtin.file_edit import FileEditTool
from Tool.builtin.file_write import FileWriteTool, _WorkspaceWriteTool
from Tool.builtin.filesystem import FileReadTool, ListTool, GlobTool, GrepTool, _WorkspaceTool
from Tool.claude_compat.wrappers import ClaudeCompatDelegatingTool
from Tool.builtin.notebook_edit import NotebookEditTool
from Tool.builtin.task_output import TaskOutputTool
from Tool.builtin.task_stop import TaskStopTool


MULTI_AGENT_TOOL_NAMES = frozenset(
    {
        "Agent",
        "AgentGet",
        "AgentList",
        "AgentWait",
        "AgentStop",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
    }
)


ChildToolFactory = Callable[[Tool, AgentRuntime], Tool | None]

# These capabilities capture parent modules. Reinstall them with child modules,
# or explicitly provide a binding factory. Do not reuse their parent bindings.
MODULE_TOOL_NAMES = frozenset({
    "skill_tool", "tool_schema_tool", "EnterPlanMode", "ExitPlanMode",
    "Config", "EnterWorktree", "ExitWorktree", "TodoWrite",
    "TaskCreate", "TaskGet", "TaskUpdate", "TaskList",
    "CodeIntelStatus", "FindDefinition", "FindReferences", "GetDocumentSymbols",
    "GetWorkspaceSymbols", "GetDiagnostics", "CodeIntelCacheStatus",
    "CodeIntelPrewarmWorkspace",
})


def apply_child_tool_factories(
    target: ToolRegistry,
    parent: ToolRegistry,
    runtime: AgentRuntime,
    factories: Mapping[str, ChildToolFactory],
) -> None:
    """Apply explicit bindings after built-in/module tool registration."""
    for name, factory in factories.items():
        original = parent.get_tool(name)
        if original is None:
            continue
        bound = factory(original, runtime)
        if bound is None:
            if target.has_tool(name):
                target.unregister_tool(name)
            continue
        if not isinstance(bound, Tool):
            raise TypeError(f"Child factory for {name!r} must return Tool or None")
        if bound.name != name:
            raise ValueError(f"Child factory for {name!r} must preserve the tool name")
        target.register_tool(bound)


def clone_tool_registry_for_workspace(
    registry: ToolRegistry,
    *,
    workspace_root: str,
    allowed_roots: Iterable[str] | None = None,
    cwd: str | None = None,
    runtime: AgentRuntime | None = None,
    tool_factories: Mapping[str, ChildToolFactory] | None = None,
    excluded_tools: Iterable[Tool] = (),
) -> ToolRegistry:
    """Build an independent registry, reusing tools that need no local binding.

    A named factory overrides inheritance: return a child-bound Tool, the same
    intentionally shared Tool, or None to omit it. Factories may not rename tools.
    Unrecognized custom tools are reused, not introspected or deep-copied; their
    owner must provide a factory if they capture mutable per-agent dependencies.
    MCP tools borrow the parent's connection; the child does not own/close it.
    """
    cloned = ToolRegistry()
    child_runtime = runtime or AgentRuntime(
        workspace_root=workspace_root,
        allowed_roots=tuple(allowed_roots or (workspace_root,)),
    )
    roots = tuple(allowed_roots or child_runtime.allowed_roots)
    workspace = dict(workspace_root=workspace_root, allowed_roots=roots, cwd=cwd or workspace_root)
    factories = dict(tool_factories or {})
    excluded_names = {tool.name for tool in excluded_tools}

    for name in registry.get_tool_names():
        tool = registry.get_tool(name)
        if name in factories:
            continue
        if (
            name in MULTI_AGENT_TOOL_NAMES or name in MODULE_TOOL_NAMES
            or name in excluded_names or "memory" in tool.get_spec().tags
        ):
            continue
        elif type(tool) is BashTool:
            register_shell_tools(
                cloned, **workspace,
                shell=tool.process_manager.shell,
                command_timeout_ms=tool.command_timeout_ms,
                max_background_tasks=tool.process_manager.max_background_tasks,
                max_output_chars=tool.max_output_chars,
                sandbox=deepcopy(tool.sandbox_config),
            )
            continue
        elif isinstance(tool, (TaskOutputTool, TaskStopTool)):
            # Registered with the child's Bash process manager, never the parent.
            continue
        elif type(tool) is FileReadTool:
            bound = FileReadTool(**workspace, max_output_chars=tool.max_output_chars)
        elif type(tool) is ListTool:
            bound = ListTool(**workspace, max_output_chars=tool.max_output_chars, max_results=tool.max_results)
        elif type(tool) is GlobTool:
            bound = GlobTool(**workspace, max_results=tool.max_results)
        elif type(tool) is GrepTool:
            bound = GrepTool(**workspace, rg_binary=tool.rg_binary,
                             max_head_limit=tool.max_head_limit, max_output_chars=tool.max_output_chars)
        elif type(tool) in (FileWriteTool, FileEditTool, NotebookEditTool):
            bound = type(tool)(**workspace)
        elif isinstance(tool, (_WorkspaceTool, _WorkspaceWriteTool, BashTool, ClaudeCompatDelegatingTool)):
            raise ValueError(
                f"Workspace-bound or delegating tool {name!r} needs a child_tool_factories binding"
            )
        else:
            bound = tool
        cloned.register_tool(bound)

    # Apply explicit bindings once, after grouped registrations (e.g. Bash).
    apply_child_tool_factories(cloned, registry, child_runtime, factories)

    # Factories cannot accidentally expose shell helpers that the parent did not
    # provide. Preserve per-registry metadata without mutating shared Tool specs.
    for name in cloned.get_tool_names():
        if not registry.has_tool(name) or (name in excluded_names and name not in factories):
            cloned.unregister_tool(name)
            continue
        cloned.register_tool(
            cloned.get_tool(name),
            visibility=registry.get_tool_visibility(name) or "resident",
            expose_in_deferred=registry.get_tool_spec(name).expose_in_deferred,
            permission_descriptor=deepcopy(registry.get_permission_descriptor(name)),
        )
    return cloned
