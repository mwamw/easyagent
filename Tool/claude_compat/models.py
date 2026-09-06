"""Pydantic models mirroring Claude Code tool input contracts."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ClaudeAgentInput(BaseModel):
    description: str = Field(description="A short (3-5 word) description of the task")
    prompt: str = Field(description="The complete task for the subagent")
    subagent_type: Optional[str] = Field(default=None, description="Subagent type")
    model: Optional[str] = Field(default=None, description="Model ID or alias for the subagent")
    run_in_background: bool = Field(default=False, description="Run in the background")
    name: Optional[str] = Field(default=None, description="Subagent name")
    team_name: Optional[str] = Field(default=None, description="Team name")
    mode: Optional[Literal["acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"]] = Field(
        default=None,
        description="Subagent permission mode",
    )
    isolation: Optional[Literal["worktree"]] = Field(default=None, description="Isolation mode")


class ClaudeAgentGetInput(BaseModel):
    agent_id: str = Field(description="Subagent ID")


class ClaudeAgentListInput(BaseModel):
    status: Optional[str] = Field(default=None, description="Filter by status")
    team_id: Optional[str] = Field(default=None, description="Filter by team ID or name")
    current_task_id: Optional[str] = Field(default=None, description="Filter by current task ID")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum number of results")


class ClaudeAgentWaitInput(BaseModel):
    agent_id: str = Field(description="Subagent ID")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="Wait timeout in milliseconds")


class ClaudeAgentStopInput(BaseModel):
    agent_id: str = Field(description="Subagent ID")
    reason: str = Field(default="", description="Reason for stopping")
    wait: bool = Field(default=False, description="Wait for the subagent to reach a terminal state")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="Wait timeout in milliseconds")


class ClaudeBashInput(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: Optional[int] = Field(default=None, description="Timeout in milliseconds")
    description: Optional[str] = Field(default=None, description="Description of what the command does")
    run_in_background: bool = Field(default=False, description="Run in the background")
    dangerouslyDisableSandbox: bool = Field(default=False, description="Request execution outside the sandbox, subject to runtime policy")


class ClaudeTaskOutputInput(BaseModel):
    task_id: str = Field(description="Background task ID")
    block: bool = Field(default=False, description="Whether to wait for completion")
    timeout: int = Field(default=0, description="Blocking wait timeout in milliseconds")


class ClaudeExitPlanModePrompt(BaseModel):
    tool: Literal["Bash"] = Field(description="Tool associated with the requested permission")
    prompt: str = Field(description="Description of the requested permission category")


class ClaudeExitPlanModeInput(BaseModel):
    allowedPrompts: list[ClaudeExitPlanModePrompt] = Field(
        default_factory=list,
        description="Permissions requested when exiting plan mode",
    )


class ClaudeEnterPlanModeInput(BaseModel):
    reason: str = Field(default="", description="Reason for entering plan mode")
    allowedActions: list[str] = Field(default_factory=list, description="Action categories allowed during planning")


class ClaudeFileEditInput(BaseModel):
    file_path: str = Field(description="Path of the file to edit")
    old_string: str = Field(description="The exact text to replace")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace all occurrences")


class ClaudeFileReadInput(BaseModel):
    file_path: str = Field(description="File path to read. Relative paths are supported; omit :line suffixes and Markdown wrappers.")
    offset: Optional[int] = Field(default=None, description="Starting line number (1-based)")
    limit: Optional[int] = Field(default=None, description="Number of lines to read")
    pages: Optional[str] = Field(default=None, description="PDF page range, for example 1-5")


class ClaudeFileWriteInput(BaseModel):
    file_path: str = Field(description="Target file path. Relative paths are supported; omit :line suffixes and Markdown wrappers.")
    content: str = Field(description="Content to write")


class ClaudeListInput(BaseModel):
    path: Optional[str] = Field(default=None, description="Directory to list; defaults to the current working directory")
    recursive: bool = Field(default=False, description="List subdirectories recursively")
    max_depth: Optional[int] = Field(default=None, ge=0, description="Maximum recursion depth; 0 lists only the current directory")
    include_hidden: bool = Field(default=True, description="Include hidden files and directories")
    directories_only: bool = Field(default=False, description="Return directories only")
    limit: int = Field(default=500, ge=1, le=5000, description="Maximum number of directory entries")


class ClaudeGlobInput(BaseModel):
    pattern: str = Field(description="Shell-style glob pattern, not a regex. Use ** explicitly for recursive matching.")
    path: Optional[str] = Field(default=None, description="Search directory; defaults to the current working directory")


class ClaudeGrepInput(BaseModel):
    pattern: str = Field(description="Regular expression for content matching, not a file glob")
    path: Optional[str] = Field(default=None, description="Search path; defaults to the current working directory")
    glob: Optional[str] = Field(default=None, description="File filter glob, for example **/*.py")
    output_mode: Optional[Literal["content", "files_with_matches", "count"]] = Field(
        default="files_with_matches",
        description="Output mode",
    )
    before_context: Optional[int] = Field(default=None, alias="-B", description="Lines of context before a match")
    after_context: Optional[int] = Field(default=None, alias="-A", description="Lines of context after a match")
    full_context: Optional[int] = Field(default=None, alias="-C", description="Lines of context around a match")
    context: Optional[int] = Field(default=None, description="Alias for context lines")
    line_numbers: Optional[bool] = Field(default=True, alias="-n", description="Show line numbers")
    ignore_case: Optional[bool] = Field(default=None, alias="-i", description="Ignore case")
    type: Optional[str] = Field(default=None, description="Ripgrep file type")
    head_limit: Optional[int] = Field(default=None, description="Maximum number of results")
    offset: Optional[int] = Field(default=None, description="Number of results to skip")
    multiline: bool = Field(default=False, description="Enable multiline matching")

    model_config = {
        "populate_by_name": True,
    }


class ClaudeTaskStopInput(BaseModel):
    task_id: Optional[str] = Field(default=None, description="Background task ID")
    shell_id: Optional[str] = Field(default=None, description="Legacy alias for the background task ID")


class ClaudeTaskCreateInput(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(default="", description="Task description")
    status: Literal["open", "in_progress", "blocked", "completed", "cancelled"] = Field(
        default="open",
        description="Task status",
    )
    owner: Optional[str] = Field(default=None, description="Task owner")
    parent_task_id: Optional[str] = Field(default=None, description="Parent task ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClaudeTaskGetInput(BaseModel):
    task_id: str = Field(description="Task ID")


class ClaudeTaskUpdateInput(BaseModel):
    task_id: str = Field(description="Task ID")
    title: Optional[str] = Field(default=None, description="New title")
    description: Optional[str] = Field(default=None, description="New description")
    status: Optional[Literal["open", "in_progress", "blocked", "completed", "cancelled"]] = Field(
        default=None,
        description="New status",
    )
    owner: Optional[str] = Field(default=None, description="New owner")
    parent_task_id: Optional[str] = Field(default=None, description="Parent task ID")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Metadata fields to merge")


class ClaudeTaskListInput(BaseModel):
    status: Optional[Literal["open", "in_progress", "blocked", "completed", "cancelled"]] = Field(
        default=None,
        description="Filter by status",
    )
    owner: Optional[str] = Field(default=None, description="Filter by owner")
    parent_task_id: Optional[str] = Field(default=None, description="Filter by parent task ID")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum number of results")


class ClaudeSendMessageInput(BaseModel):
    recipient_type: Literal["agent", "team", "task"] = Field(description="Recipient type")
    recipient_id: str = Field(description="Recipient ID or team name")
    content: str = Field(description="Message content to send")
    sender_id: Optional[str] = Field(default=None, description="Sender identity")
    ttl_ms: Optional[int] = Field(default=None, ge=0, description="Message time to live in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClaudeTeamCreateInput(BaseModel):
    name: str = Field(description="Team name")
    description: str = Field(default="", description="Team description")
    member_agent_ids: list[str] = Field(default_factory=list, description="Initial member agent IDs")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClaudeTeamDeleteInput(BaseModel):
    team_id: str = Field(description="Team ID or name")


class ClaudeMailboxReadInput(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="Agent ID; defaults to the current agent")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum number of results")
    include_consumed: bool = Field(default=False, description="Include consumed messages")
    include_expired: bool = Field(default=False, description="Include expired messages")
    mark_delivered: bool = Field(default=True, description="Mark queued messages as delivered")


class ClaudeMailboxAckInput(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="Agent ID; defaults to the current agent")
    message_ids: list[str] = Field(default_factory=list, description="Message IDs to acknowledge as consumed")
    ack_all: bool = Field(default=False, description="Acknowledge all unconsumed messages for the current agent")


class ClaudeListMcpResourcesInput(BaseModel):
    server: Optional[str] = Field(default=None, description="MCP server name")


class ClaudeNotebookEditInput(BaseModel):
    notebook_path: str = Field(description="Notebook path. Relative paths are supported; omit :line suffixes and Markdown wrappers.")
    cell_id: Optional[str] = Field(default=None, description="Cell ID")
    new_source: str = Field(description="New cell source")
    cell_type: Optional[Literal["code", "markdown"]] = Field(default=None, description="Cell type")
    edit_mode: Optional[Literal["replace", "insert", "delete"]] = Field(default="replace", description="Edit mode")


class ClaudeReadMcpResourceInput(BaseModel):
    server: str = Field(description="MCP server name")
    uri: str = Field(description="Resource URI")


class ClaudeTodoItem(BaseModel):
    content: str = Field(description="Task content")
    status: Literal["pending", "in_progress", "completed"] = Field(description="Task status")
    activeForm: str = Field(description="Present-progressive description of the active task")


class ClaudeTodoWriteInput(BaseModel):
    todos: list[ClaudeTodoItem] = Field(default_factory=list, description="The complete todo list")


class ClaudeWebFetchInput(BaseModel):
    url: str = Field(description="URL to fetch")
    prompt: str = Field(description="What to extract from the fetched content")


class ClaudeWebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    allowed_domains: list[str] = Field(default_factory=list, description="Domains to include")
    blocked_domains: list[str] = Field(default_factory=list, description="Domains to exclude")


class ClaudeAskUserOption(BaseModel):
    label: str = Field(description="Option label")
    description: str = Field(description="Option description")
    preview: Optional[str] = Field(default=None, description="Optional preview")


class ClaudeAskUserQuestion(BaseModel):
    question: str = Field(description="Complete question")
    header: str = Field(description="Short label")
    options: list[ClaudeAskUserOption] = Field(min_length=2, max_length=4, description="Available options")
    multiSelect: bool = Field(default=False, description="Allow multiple selections")


class ClaudeAskUserQuestionInput(BaseModel):
    questions: list[ClaudeAskUserQuestion] = Field(min_length=1, max_length=4, description="Questions to ask")
    source: Optional[str] = Field(default=None, description="Question source")


class ClaudeConfigInput(BaseModel):
    setting: str = Field(description="Configuration key")
    value: Optional[str] = Field(default=None, description="New configuration value; complex values may be encoded as JSON strings")


class ClaudeEnterWorktreeInput(BaseModel):
    name: Optional[str] = Field(default=None, description="Worktree name")


class ClaudeExitWorktreeInput(BaseModel):
    action: Literal["keep", "remove"] = Field(description="Exit action")
    discard_changes: bool = Field(default=False, description="Discard changes when removing the worktree")


class ClaudeToolCompatibilityEnvelope(BaseModel):
    """Serializable catalog and manifest envelope."""

    tool_name: str = Field(description="Claude-style tool name")
    parameters_model_name: str = Field(description="Parameter model name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
