"""Catalog of Claude Code compatible tool contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from pydantic import BaseModel

from .models import (
    ClaudeAgentInput,
    ClaudeAgentGetInput,
    ClaudeAgentListInput,
    ClaudeAgentStopInput,
    ClaudeAgentWaitInput,
    ClaudeAskUserQuestionInput,
    ClaudeBashInput,
    ClaudeConfigInput,
    ClaudeEnterPlanModeInput,
    ClaudeEnterWorktreeInput,
    ClaudeExitPlanModeInput,
    ClaudeExitWorktreeInput,
    ClaudeFileEditInput,
    ClaudeFileReadInput,
    ClaudeFileWriteInput,
    ClaudeGlobInput,
    ClaudeGrepInput,
    ClaudeListMcpResourcesInput,
    ClaudeMailboxAckInput,
    ClaudeMailboxReadInput,
    ClaudeNotebookEditInput,
    ClaudeReadMcpResourceInput,
    ClaudeSendMessageInput,
    ClaudeTaskOutputInput,
    ClaudeTaskCreateInput,
    ClaudeTaskGetInput,
    ClaudeTaskListInput,
    ClaudeTaskStopInput,
    ClaudeTaskUpdateInput,
    ClaudeTeamCreateInput,
    ClaudeTeamDeleteInput,
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
    ClaudeToolDefinition("AgentGet", ClaudeAgentGetInput, "查询单个子 agent 运行时状态", read_only=True, tags=("agent", "runtime")),
    ClaudeToolDefinition("AgentList", ClaudeAgentListInput, "列出当前 runtime 中的子 agent", read_only=True, tags=("agent", "runtime")),
    ClaudeToolDefinition("AgentWait", ClaudeAgentWaitInput, "等待子 agent 进入当前可观察状态", read_only=True, tags=("agent", "runtime")),
    ClaudeToolDefinition("AgentStop", ClaudeAgentStopInput, "请求停止子 agent", destructive=True, tags=("agent", "runtime")),
    ClaudeToolDefinition("Bash", ClaudeBashInput, "Claude Code shell 命令执行工具", destructive=True, tags=("shell", "local")),
    ClaudeToolDefinition("TaskOutput", ClaudeTaskOutputInput, "读取后台任务输出", read_only=True, tags=("shell", "background")),
    ClaudeToolDefinition("TaskCreate", ClaudeTaskCreateInput, "创建结构化任务", tags=("planning", "task")),
    ClaudeToolDefinition("TaskGet", ClaudeTaskGetInput, "读取单个结构化任务", read_only=True, tags=("planning", "task")),
    ClaudeToolDefinition("TaskUpdate", ClaudeTaskUpdateInput, "更新结构化任务", tags=("planning", "task")),
    ClaudeToolDefinition("TaskList", ClaudeTaskListInput, "列出结构化任务", read_only=True, tags=("planning", "task")),
    ClaudeToolDefinition("SendMessage", ClaudeSendMessageInput, "向 agent 或团队发送结构化消息", tags=("agent", "team")),
    ClaudeToolDefinition("MailboxRead", ClaudeMailboxReadInput, "读取 agent mailbox 中的消息", tags=("agent", "mailbox")),
    ClaudeToolDefinition("MailboxAck", ClaudeMailboxAckInput, "确认 mailbox 消息已被消费", tags=("agent", "mailbox")),
    ClaudeToolDefinition("TeamCreate", ClaudeTeamCreateInput, "创建 agent 团队", tags=("agent", "team")),
    ClaudeToolDefinition("TeamDelete", ClaudeTeamDeleteInput, "删除 agent 团队", tags=("agent", "team")),
    ClaudeToolDefinition("EnterPlanMode", ClaudeEnterPlanModeInput, "进入 plan 模式", tags=("plan",)),
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
