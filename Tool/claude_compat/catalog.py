"""Catalog of Claude Code compatible tool contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from pydantic import BaseModel

from .models import (
    ClaudeAgentInput,
    ClaudeAskUserQuestionInput,
    ClaudeBashInput,
    ClaudeConfigInput,
    ClaudeEnterWorktreeInput,
    ClaudeExitPlanModeInput,
    ClaudeExitWorktreeInput,
    ClaudeFileEditInput,
    ClaudeFileReadInput,
    ClaudeFileWriteInput,
    ClaudeGlobInput,
    ClaudeGrepInput,
    ClaudeListMcpResourcesInput,
    ClaudeNotebookEditInput,
    ClaudeReadMcpResourceInput,
    ClaudeTaskOutputInput,
    ClaudeTaskStopInput,
    ClaudeTodoWriteInput,
    ClaudeWebFetchInput,
    ClaudeWebSearchInput,
)


@dataclass(frozen=True, slots=True)
class ClaudeToolDefinition:
    name: str
    parameters_model: Type[BaseModel]
    description: str
    read_only: bool = False
    destructive: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


_CLAUDE_TOOL_DEFINITIONS = [
    ClaudeToolDefinition("Agent", ClaudeAgentInput, "Claude Code 子 agent 委派工具", tags=("agent", "orchestration")),
    ClaudeToolDefinition("Bash", ClaudeBashInput, "Claude Code shell 命令执行工具", destructive=True, tags=("shell", "local")),
    ClaudeToolDefinition("TaskOutput", ClaudeTaskOutputInput, "读取后台任务输出", read_only=True, tags=("shell", "background")),
    ClaudeToolDefinition("ExitPlanMode", ClaudeExitPlanModeInput, "退出 plan 模式", tags=("plan",)),
    ClaudeToolDefinition("FileEdit", ClaudeFileEditInput, "精确编辑文件内容", destructive=True, tags=("filesystem", "edit")),
    ClaudeToolDefinition("FileRead", ClaudeFileReadInput, "读取文件或 PDF 内容", read_only=True, tags=("filesystem", "read")),
    ClaudeToolDefinition("FileWrite", ClaudeFileWriteInput, "写入或覆盖文件", destructive=True, tags=("filesystem", "write")),
    ClaudeToolDefinition("Glob", ClaudeGlobInput, "按 glob 模式查找文件", read_only=True, tags=("filesystem", "search")),
    ClaudeToolDefinition("Grep", ClaudeGrepInput, "按正则检索文件内容", read_only=True, tags=("filesystem", "search")),
    ClaudeToolDefinition("TaskStop", ClaudeTaskStopInput, "停止后台任务", destructive=True, tags=("shell", "background")),
    ClaudeToolDefinition("ListMcpResources", ClaudeListMcpResourcesInput, "列出 MCP 资源", read_only=True, tags=("mcp", "resource")),
    ClaudeToolDefinition("NotebookEdit", ClaudeNotebookEditInput, "编辑 Jupyter Notebook", destructive=True, tags=("notebook",)),
    ClaudeToolDefinition("ReadMcpResource", ClaudeReadMcpResourceInput, "读取 MCP 资源内容", read_only=True, tags=("mcp", "resource")),
    ClaudeToolDefinition("TodoWrite", ClaudeTodoWriteInput, "维护 todo 列表", tags=("planning", "todo")),
    ClaudeToolDefinition("WebFetch", ClaudeWebFetchInput, "抓取网页并提炼正文", read_only=True, tags=("web", "fetch")),
    ClaudeToolDefinition("WebSearch", ClaudeWebSearchInput, "搜索公开网页资料", read_only=True, tags=("web", "search")),
    ClaudeToolDefinition("AskUserQuestion", ClaudeAskUserQuestionInput, "结构化向用户提问", tags=("interaction",)),
    ClaudeToolDefinition("Config", ClaudeConfigInput, "读取或修改运行配置", tags=("config",)),
    ClaudeToolDefinition("EnterWorktree", ClaudeEnterWorktreeInput, "进入隔离 git worktree", destructive=True, tags=("git", "worktree")),
    ClaudeToolDefinition("ExitWorktree", ClaudeExitWorktreeInput, "退出或移除 git worktree", destructive=True, tags=("git", "worktree")),
]

CLAUDE_TOOL_ORDER = tuple(definition.name for definition in _CLAUDE_TOOL_DEFINITIONS)
CLAUDE_TOOL_MODELS = {
    definition.name: definition.parameters_model
    for definition in _CLAUDE_TOOL_DEFINITIONS
}
_CLAUDE_TOOL_BY_NAME = {
    definition.name: definition
    for definition in _CLAUDE_TOOL_DEFINITIONS
}


def get_claude_tool_definition(name: str) -> ClaudeToolDefinition:
    try:
        return _CLAUDE_TOOL_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(CLAUDE_TOOL_ORDER)
        raise KeyError(f"Unknown Claude tool '{name}'. Available: {available}") from exc
