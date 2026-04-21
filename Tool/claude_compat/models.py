"""Pydantic models mirroring Claude Code tool input contracts."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ClaudeAgentInput(BaseModel):
    description: str = Field(description="3-5 个词的任务概述")
    prompt: str = Field(description="交给子 agent 的完整任务描述")
    subagent_type: Optional[str] = Field(default=None, description="子 agent 类型")
    model: Optional[Literal["sonnet", "opus", "haiku"]] = Field(default=None, description="模型覆盖")
    run_in_background: bool = Field(default=False, description="是否后台运行")
    name: Optional[str] = Field(default=None, description="子 agent 名称")
    team_name: Optional[str] = Field(default=None, description="团队名称")
    mode: Optional[Literal["acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"]] = Field(
        default=None,
        description="子 agent 权限模式",
    )
    isolation: Optional[Literal["worktree"]] = Field(default=None, description="隔离模式")


class ClaudeAgentGetInput(BaseModel):
    agent_id: str = Field(description="子 agent ID")


class ClaudeAgentListInput(BaseModel):
    status: Optional[str] = Field(default=None, description="状态过滤")
    team_id: Optional[str] = Field(default=None, description="团队 ID 或名称过滤")
    current_task_id: Optional[str] = Field(default=None, description="当前任务 ID 过滤")
    limit: int = Field(default=100, ge=1, le=500, description="返回数量上限")


class ClaudeAgentWaitInput(BaseModel):
    agent_id: str = Field(description="子 agent ID")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="等待超时时间，毫秒")


class ClaudeAgentStopInput(BaseModel):
    agent_id: str = Field(description="子 agent ID")
    reason: str = Field(default="", description="停止原因")
    wait: bool = Field(default=False, description="是否等待子 agent 进入终态")
    timeout_ms: Optional[int] = Field(default=None, ge=0, description="等待超时时间，毫秒")


class ClaudeBashInput(BaseModel):
    command: str = Field(description="要执行的 shell 命令")
    timeout: Optional[int] = Field(default=None, description="超时时间，毫秒")
    description: Optional[str] = Field(default=None, description="命令用途说明")
    run_in_background: bool = Field(default=False, description="是否后台运行")
    dangerouslyDisableSandbox: bool = Field(default=False, description="是否禁用沙箱")


class ClaudeTaskOutputInput(BaseModel):
    task_id: str = Field(description="后台任务 ID")
    block: bool = Field(default=False, description="是否阻塞等待")
    timeout: int = Field(default=0, description="阻塞等待超时时间，毫秒")


class ClaudeExitPlanModePrompt(BaseModel):
    tool: Literal["Bash"] = Field(description="权限提示关联的工具")
    prompt: str = Field(description="权限类别描述")


class ClaudeExitPlanModeInput(BaseModel):
    allowedPrompts: list[ClaudeExitPlanModePrompt] = Field(
        default_factory=list,
        description="退出计划模式时声明的权限类别",
    )


class ClaudeEnterPlanModeInput(BaseModel):
    reason: str = Field(default="", description="进入 plan 模式的原因")
    allowedActions: list[str] = Field(default_factory=list, description="计划阶段允许的动作类别")


class ClaudeFileEditInput(BaseModel):
    file_path: str = Field(description="要修改的绝对路径")
    old_string: str = Field(description="待替换文本")
    new_string: str = Field(description="替换后的文本")
    replace_all: bool = Field(default=False, description="是否替换所有匹配项")


class ClaudeFileReadInput(BaseModel):
    file_path: str = Field(description="要读取的文件路径")
    offset: Optional[int] = Field(default=None, description="起始行号")
    limit: Optional[int] = Field(default=None, description="读取行数")
    pages: Optional[str] = Field(default=None, description="PDF 页范围")


class ClaudeFileWriteInput(BaseModel):
    file_path: str = Field(description="目标绝对路径")
    content: str = Field(description="写入内容")


class ClaudeGlobInput(BaseModel):
    pattern: str = Field(description="glob 匹配模式")
    path: Optional[str] = Field(default=None, description="搜索目录")


class ClaudeGrepInput(BaseModel):
    pattern: str = Field(description="正则匹配模式")
    path: Optional[str] = Field(default=None, description="搜索路径")
    glob: Optional[str] = Field(default=None, description="文件过滤 glob")
    output_mode: Optional[Literal["content", "files_with_matches", "count"]] = Field(
        default="files_with_matches",
        description="输出模式",
    )
    before_context: Optional[int] = Field(default=None, alias="-B", description="前文行数")
    after_context: Optional[int] = Field(default=None, alias="-A", description="后文行数")
    full_context: Optional[int] = Field(default=None, alias="-C", description="上下文行数")
    context: Optional[int] = Field(default=None, description="上下文行数别名")
    line_numbers: Optional[bool] = Field(default=True, alias="-n", description="是否显示行号")
    ignore_case: Optional[bool] = Field(default=None, alias="-i", description="是否忽略大小写")
    type: Optional[str] = Field(default=None, description="ripgrep 文件类型")
    head_limit: Optional[int] = Field(default=None, description="返回条数上限")
    offset: Optional[int] = Field(default=None, description="跳过条数")
    multiline: bool = Field(default=False, description="是否开启 multiline")

    model_config = {
        "populate_by_name": True,
    }


class ClaudeTaskStopInput(BaseModel):
    task_id: Optional[str] = Field(default=None, description="后台任务 ID")
    shell_id: Optional[str] = Field(default=None, description="旧版 shell ID")


class ClaudeTaskCreateInput(BaseModel):
    title: str = Field(description="任务标题")
    description: str = Field(default="", description="任务描述")
    status: Literal["open", "in_progress", "blocked", "completed", "cancelled"] = Field(
        default="open",
        description="任务状态",
    )
    owner: Optional[str] = Field(default=None, description="任务归属者")
    parent_task_id: Optional[str] = Field(default=None, description="父任务 ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class ClaudeTaskGetInput(BaseModel):
    task_id: str = Field(description="任务 ID")


class ClaudeTaskUpdateInput(BaseModel):
    task_id: str = Field(description="任务 ID")
    title: Optional[str] = Field(default=None, description="新标题")
    description: Optional[str] = Field(default=None, description="新描述")
    status: Optional[Literal["open", "in_progress", "blocked", "completed", "cancelled"]] = Field(
        default=None,
        description="新状态",
    )
    owner: Optional[str] = Field(default=None, description="新归属者")
    parent_task_id: Optional[str] = Field(default=None, description="父任务 ID")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="元数据增量")


class ClaudeTaskListInput(BaseModel):
    status: Optional[Literal["open", "in_progress", "blocked", "completed", "cancelled"]] = Field(
        default=None,
        description="状态过滤",
    )
    owner: Optional[str] = Field(default=None, description="归属者过滤")
    parent_task_id: Optional[str] = Field(default=None, description="父任务 ID 过滤")
    limit: int = Field(default=100, ge=1, le=500, description="返回数量上限")


class ClaudeSendMessageInput(BaseModel):
    recipient_type: Literal["agent", "team", "task"] = Field(description="接收方类型")
    recipient_id: str = Field(description="接收方 ID 或团队名称")
    content: str = Field(description="要发送的消息内容")
    sender_id: Optional[str] = Field(default=None, description="发送方标识")
    ttl_ms: Optional[int] = Field(default=None, ge=0, description="消息 TTL，毫秒")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class ClaudeTeamCreateInput(BaseModel):
    name: str = Field(description="团队名称")
    description: str = Field(default="", description="团队描述")
    member_agent_ids: list[str] = Field(default_factory=list, description="初始成员 agent ID 列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class ClaudeTeamDeleteInput(BaseModel):
    team_id: str = Field(description="团队 ID，也支持团队名称")


class ClaudeMailboxReadInput(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="agent ID，不填则默认当前 agent")
    limit: int = Field(default=100, ge=1, le=500, description="返回数量上限")
    include_consumed: bool = Field(default=False, description="是否包含已消费消息")
    include_expired: bool = Field(default=False, description="是否包含已过期消息")
    mark_delivered: bool = Field(default=True, description="是否把 queued 消息标记为 delivered")


class ClaudeMailboxAckInput(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="agent ID，不填则默认当前 agent")
    message_ids: list[str] = Field(default_factory=list, description="待确认消费的消息 ID")
    ack_all: bool = Field(default=False, description="是否确认当前 agent 所有未消费消息")


class ClaudeListMcpResourcesInput(BaseModel):
    server: Optional[str] = Field(default=None, description="MCP server 名称")


class ClaudeNotebookEditInput(BaseModel):
    notebook_path: str = Field(description="Notebook 文件路径")
    cell_id: Optional[str] = Field(default=None, description="单元格 ID")
    new_source: str = Field(description="新的单元格内容")
    cell_type: Optional[Literal["code", "markdown"]] = Field(default=None, description="单元格类型")
    edit_mode: Optional[Literal["replace", "insert", "delete"]] = Field(default="replace", description="编辑模式")


class ClaudeReadMcpResourceInput(BaseModel):
    server: str = Field(description="MCP server 名称")
    uri: str = Field(description="资源 URI")


class ClaudeTodoItem(BaseModel):
    content: str = Field(description="todo 内容")
    status: Literal["pending", "in_progress", "completed"] = Field(description="todo 状态")
    activeForm: str = Field(description="当前进行式描述")


class ClaudeTodoWriteInput(BaseModel):
    todos: list[ClaudeTodoItem] = Field(default_factory=list, description="完整 todo 列表")


class ClaudeWebFetchInput(BaseModel):
    url: str = Field(description="要抓取的 URL")
    prompt: str = Field(description="对抓取内容的处理提示")


class ClaudeWebSearchInput(BaseModel):
    query: str = Field(description="搜索词")
    allowed_domains: list[str] = Field(default_factory=list, description="允许域名白名单")
    blocked_domains: list[str] = Field(default_factory=list, description="禁止域名黑名单")


class ClaudeAskUserOption(BaseModel):
    label: str = Field(description="选项标签")
    description: str = Field(description="选项说明")
    preview: Optional[str] = Field(default=None, description="选项预览")


class ClaudeAskUserQuestion(BaseModel):
    question: str = Field(description="完整问题")
    header: str = Field(description="短标签")
    options: list[ClaudeAskUserOption] = Field(min_length=2, max_length=4, description="候选选项")
    multiSelect: bool = Field(default=False, description="是否多选")


class ClaudeAskUserQuestionInput(BaseModel):
    questions: list[ClaudeAskUserQuestion] = Field(min_length=1, max_length=4, description="问题列表")
    source: Optional[str] = Field(default=None, description="问题来源")


class ClaudeConfigInput(BaseModel):
    setting: str = Field(description="配置 key")
    value: Optional[str] = Field(default=None, description="新的配置值(如果是复杂类型可以转字符串传入)")


class ClaudeEnterWorktreeInput(BaseModel):
    name: Optional[str] = Field(default=None, description="worktree 名称")


class ClaudeExitWorktreeInput(BaseModel):
    action: Literal["keep", "remove"] = Field(description="退出行为")
    discard_changes: bool = Field(default=False, description="是否丢弃改动")


class ClaudeToolCompatibilityEnvelope(BaseModel):
    """便于序列化 catalog/manifest 的统一结构。"""

    tool_name: str = Field(description="Claude 风格工具名")
    parameters_model_name: str = Field(description="参数模型名称")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元信息")
