"""Claude-style Bash tool backed by ProcessManager."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from pydantic import BaseModel

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeBashInput
from ..runtime import BackgroundTaskSnapshot, FilesystemAccessError, FilesystemGuard, PathResolutionError, ProcessManager
from .input_normalization import normalize_generic_input
from .bash_sandbox import (
    BashSandboxConfig,
    BashSandboxError,
    LinuxBubblewrapSandbox,
    SandboxUnavailableError,
)


DEFAULT_BASH_OUTPUT_CHARS = 120000

BASH_TOOL_PROMPT = """Executes a given bash command and returns its output.

Commands run in the configured working directory and shell. Shell state is not preserved between calls.

IMPORTANT: Avoid using this tool to run find, grep, cat, head, tail, sed, awk, or echo commands, unless explicitly instructed or after verifying that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool:
- File search: Use Glob.
- Content search: Use Grep.
- Read files: Use FileRead.
- Edit files: Use FileEdit.
- Write files: Use FileWrite.
- Communication: Output text directly, not through echo or printf.

While Bash can do similar things, the built-in tools make it easier to review tool calls and give permission.

# Instructions
- If your command will create directories or files, verify the parent directory exists and is the correct location.
- Always quote file paths that contain spaces with double quotes.
- Try to maintain your current working directory throughout the session by using absolute paths and avoiding cd. You may use cd if the user explicitly requests it.
- You may specify an optional timeout in milliseconds. If omitted, the configured command timeout applies.
- When issuing multiple commands:
  - If the commands are independent and can run in parallel, make multiple Bash tool calls in a single message. Example: if you need to run "git status" and "git diff", send a single message with two Bash tool calls.
  - If commands depend on each other and must run sequentially, use a single Bash call with && to chain them together.
  - Use ; only when commands must run sequentially but later commands need not depend on earlier success.
  - Do not use newlines to separate commands; newlines inside quoted strings are fine.
- For git commands:
  - Prefer creating a new commit rather than amending an existing commit.
  - Before destructive operations (git reset --hard, git push --force, git checkout --), consider a safer alternative.
  - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign, -c commit.gpgsign=false) unless explicitly asked. If a hook fails, investigate and fix it.
  - Never commit or push unless the user requests it.
  - Review git status and git diff before staging. Avoid committing secrets or unrelated user changes.
  - If a commit fails because of a hook, fix the issue and create a new commit; do not amend a previous commit.
- Avoid unnecessary sleep commands:
  - Do not sleep between commands that can run immediately.
  - Do not retry failing commands in a sleep loop — diagnose the root cause.
  - For long-running commands, set run_in_background=true and use TaskOutput to obtain output and completion status.
  - A background task handle means the command started, not that it succeeded. Check its exit status.
  - Use TaskStop when a background task must be terminated.

Results include task_id, status, return_code, stdout, stderr, and truncation information. Inspect failures and truncated output before deciding on the next action.

# Committing changes with git

Only create commits when requested by the user. If unclear, ask first. When the user asks you to create a new git commit, follow these steps carefully:

You can call multiple tools in a single response. When multiple independent pieces of information are requested and all commands are likely to succeed, run multiple tool calls in parallel for optimal performance. The numbered steps below indicate which commands should be batched in parallel.

Git Safety Protocol:
- NEVER update the git config.
- NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless the user explicitly requests these actions. Taking unauthorized destructive actions is unhelpful and can result in lost work, so it's best to ONLY run these commands when given direct instructions.
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it.
- NEVER run force push to main/master, warn the user if they request it.
- CRITICAL: Always create NEW commits rather than amending, unless the user explicitly requests a git amend. When a pre-commit hook fails, the commit did NOT happen -- so --amend would modify the PREVIOUS commit, which may result in destroying work or losing previous changes. Instead, after hook failure, fix the issue, re-stage, and create a NEW commit.
- When staging files, prefer adding specific files by name rather than using "git add -A" or "git add .", which can accidentally include sensitive files (.env, credentials) or large binaries.
- NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.

1. Run the following bash commands in parallel, each using the Bash tool:
   - Run a git status command to see all untracked files. IMPORTANT: Never use the -uall flag as it can cause memory issues on large repos.
   - Run a git diff command to see both staged and unstaged changes that will be committed.
   - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style.
2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
   - Summarize the nature of the changes (e.g. new feature, enhancement, bug fix, refactoring, test, docs). Ensure the message accurately reflects the changes and their purpose.
   - Do not commit files that likely contain secrets (.env, credentials.json, etc). Warn the user if they specifically request to commit those files.
   - Draft a concise (1-2 sentences) commit message that focuses on the "why" rather than the "what".
   - Ensure it accurately reflects the changes and their purpose.
3. Run these dependent commands sequentially:
   - Add relevant untracked files to the staging area.
   - Create the commit with the message.
   - Run git status after the commit completes to verify success.
4. If the commit fails due to a pre-commit hook: fix the issue and create a NEW commit.

Important notes:
- During this commit workflow, NEVER run additional commands to read or explore code, besides git bash commands.
- NEVER use the TodoWrite or Agent tools for this commit workflow.
- DO NOT push to the remote repository unless the user explicitly asks you to do so.
- IMPORTANT: Never use git commands with the -i flag (like git rebase -i or git add -i) since they require interactive input which is not supported.
- IMPORTANT: Do not use --no-edit with git rebase commands, as the --no-edit flag is not a valid option for git rebase.
- If there are no changes to commit (i.e., no untracked files and no modifications), do not create an empty commit.
- In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, as in this example:
<example>
git commit -m "$(cat <<'EOF'
Commit message here.
EOF
)"
</example>

# Creating pull requests

Use the gh command via the Bash tool for GitHub-related tasks including working with issues, pull requests, checks, and releases, when gh is installed and authenticated. If given a GitHub URL, use gh to get the information needed when available.

IMPORTANT: When the user asks you to create a pull request, follow these steps carefully:

1. Run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:
   - Run a git status command to see all untracked files (never use -uall).
   - Run a git diff command to see both staged and unstaged changes that will be committed.
   - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote.
   - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch).
2. Analyze all changes that will be included in the pull request, making sure to look at ALL relevant commits, NOT just the latest commit, and draft a pull request title and summary:
   - Keep the PR title short (under 70 characters).
   - Use the description/body for details, not the title.
3. Run these dependent commands sequentially:
   - Create a new branch if needed.
   - Push to remote with -u if needed and authorized.
   - Create the PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting.
<example>
gh pr create --title "the pr title" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request...]
EOF
)"
</example>

Important:
- DO NOT use the TodoWrite or Agent tools for this PR workflow.
- Return the PR URL when you're done, so the user can see it.

# Other common operations
- View comments on a GitHub PR: gh api repos/foo/bar/pulls/123/comments"""


def _normalize_workspace_root(workspace_root: Optional[str]) -> str:
    return os.path.abspath(workspace_root or os.getcwd())


def _structured_error(
    message: str,
    *,
    error_type: str,
    metadata: Optional[dict[str, Any]] = None,
    structured_data: Any = None,
) -> ToolResult:
    return ToolResult(
        status="error",
        content=message,
        error_type=error_type,
        metadata=dict(metadata or {}),
        structured_data=structured_data,
    )


def _clip_text(value: str, *, max_chars: int = DEFAULT_BASH_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    clipped = value[:max_chars].rstrip()
    return f"{clipped}\n\n...[truncated]", True


def _stringify_command(command: Any) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def _snapshot_to_dict(snapshot: BackgroundTaskSnapshot) -> dict[str, Any]:
    return {
        "task_id": snapshot.task_id,
        "command": snapshot.command,
        "cwd": snapshot.cwd,
        "status": snapshot.status,
        "return_code": snapshot.return_code,
        "stdout": snapshot.stdout,
        "stderr": snapshot.stderr,
        "started_at": snapshot.started_at,
        "finished_at": snapshot.finished_at,
    }


def _format_background_snapshot(snapshot: BackgroundTaskSnapshot, *, max_output_chars: int) -> tuple[str, bool]:
    sections = [
        f"任务 ID: {snapshot.task_id}",
        f"状态: {snapshot.status}",
        f"命令: {_stringify_command(snapshot.command)}",
        f"cwd: {snapshot.cwd}",
    ]
    if snapshot.return_code is not None:
        sections.append(f"退出码: {snapshot.return_code}")
    if snapshot.stdout:
        sections.append(f"stdout:\n{snapshot.stdout}")
    if snapshot.stderr:
        sections.append(f"stderr:\n{snapshot.stderr}")
    display_text = "\n\n".join(sections)
    return _clip_text(display_text, max_chars=max_output_chars)


class _ShellToolBase(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: type[BaseModel],
        workspace_root: Optional[str] = None,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        shell: str = "bash",
        command_timeout_ms: int = 120000,
        max_background_tasks: int = 8,
        max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
        process_manager: Optional[ProcessManager] = None,
        guidance: str = "",
        prompt: str = "",
        read_only: bool = False,
        destructive: bool = True,
        supports_parallel: bool = True,
        tags: Optional[list[str]] = None,
        resource_scope: Optional[list[str]] = None,
    ):
        self.workspace_root = _normalize_workspace_root(workspace_root)
        self.guard = FilesystemGuard(self.workspace_root, allowed_roots=allowed_roots)
        self.cwd = self.guard.resolve_directory(cwd or self.workspace_root, cwd=self.workspace_root, must_exist=True)
        self.max_output_chars = max_output_chars
        self.command_timeout_ms = command_timeout_ms
        self.process_manager = process_manager or ProcessManager(shell=shell, max_background_tasks=max_background_tasks)
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            guidance=guidance,
            prompt=prompt,
            read_only=read_only,
            destructive=destructive,
            supports_parallel=supports_parallel,
            source="builtin",
            tags=list(tags or []),
            side_effect_level="high" if destructive else ("none" if read_only else "medium"),
            resource_scope=list(resource_scope or ["process", "filesystem", "workspace"]),
        )

    def _tool_error(
        self,
        message: str,
        *,
        error_type: str = "tool_error",
        metadata: Optional[dict[str, Any]] = None,
        structured_data: Any = None,
    ) -> ToolResult:
        return _structured_error(
            message,
            error_type=error_type,
            metadata=metadata,
            structured_data=structured_data,
        )


class BashTool(_ShellToolBase):
    """Execute shell commands in the local workspace."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        shell: str = "bash",
        command_timeout_ms: int = 120000,
        max_background_tasks: int = 8,
        max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
        process_manager: Optional[ProcessManager] = None,
        sandbox: BashSandboxConfig | None = None,
    ):
        super().__init__(
            name="Bash",
            description="Executes a shell command in the foreground or background.",
            parameters=ClaudeBashInput,
            workspace_root=workspace_root,
            allowed_roots=allowed_roots,
            cwd=cwd,
            shell=shell,
            command_timeout_ms=command_timeout_ms,
            max_background_tasks=max_background_tasks,
            max_output_chars=max_output_chars,
            process_manager=process_manager,
            guidance="Run tests, builds, scripts, and Git commands. Use TaskOutput for background results.",
            prompt=BASH_TOOL_PROMPT,
            read_only=False,
            destructive=True,
            supports_parallel=False,
            tags=["shell", "local", "claude_code"],
            resource_scope=["process", "filesystem", "workspace"],
        )
        self.sandbox_config = sandbox or BashSandboxConfig()
        self.sandbox = LinuxBubblewrapSandbox(
            self.sandbox_config,
            workspace_root=self.workspace_root,
            allowed_roots=self.guard.allowed_roots,
            shell=shell,
        )
        sandbox_status = self.sandbox.status()
        if (
            self.sandbox_config.enabled
            and not sandbox_status.available
            and (
                self.sandbox_config.fail_if_unavailable
                or not self.sandbox_config.allow_unsandboxed_commands
            )
        ):
            raise SandboxUnavailableError(
                "; ".join(sandbox_status.errors) or "sandbox unavailable"
            )
        self.process_manager.register_cleanup(self.sandbox.close)
        if self.sandbox_config.enabled:
            permission_note = (
                " Commands run inside a Linux Bubblewrap sandbox by default."
                if sandbox_status.available
                else " Sandbox was requested but is unavailable; configured fallback policy applies."
            )
            self.spec.prompt = f"{self.spec.prompt.rstrip()}\n\n# Sandbox\n{permission_note.strip()}"

    def get_sandbox_status(self) -> dict[str, object]:
        return self.sandbox.status().to_dict()

    def get_permission_extensions(self, parameters: dict[str, Any]) -> dict[str, Any]:
        command = normalize_generic_input(parameters.get("command", ""))
        dangerously_disable = bool(parameters.get("dangerouslyDisableSandbox", False))
        try:
            decision = self.sandbox.decide(
                command, dangerously_disable=dangerously_disable,
            )
            return {
                "sandbox": {
                    **decision.to_dict(),
                    "will_sandbox": decision.applied,
                    "auto_allow": decision.auto_allow,
                }
            }
        except BashSandboxError as exc:
            return {
                "sandbox": {
                    "configured": self.sandbox_config.enabled,
                    "will_sandbox": False,
                    "auto_allow": False,
                    "bypassed": dangerously_disable,
                    "error": str(exc),
                    "error_type": exc.error_type,
                }
            }

    def run(self, parameters: dict) -> ToolResult:
        command = normalize_generic_input(parameters.get("command", ""))
        description = parameters.get("description")
        run_in_background = bool(parameters.get("run_in_background", False))
        dangerously_disable_sandbox = bool(parameters.get("dangerouslyDisableSandbox", False))
        timeout_ms = parameters.get("timeout")
        timeout_ms = self.command_timeout_ms if timeout_ms is None else int(timeout_ms)

        if not command:
            return self._tool_error(
                "命令不能为空。",
                error_type="invalid_parameters",
                structured_data={"reason": "empty_command"},
            )

        try:
            sandbox_decision = self.sandbox.decide(
                command,
                dangerously_disable=dangerously_disable_sandbox,
            )
            sandbox_metadata = sandbox_decision.to_dict()
            execution_command: Any = command
            execution_env = None
            use_shell = True
            if sandbox_decision.applied:
                sandboxed = self.sandbox.wrap(command, cwd=self.cwd)
                execution_command = list(sandboxed.argv)
                execution_env = sandboxed.environment
                use_shell = False
        except BashSandboxError as exc:
            return self._tool_error(
                f"Sandbox execution rejected: {exc}",
                error_type=exc.error_type,
                metadata={"command": command, "cwd": self.cwd, "sandbox": self.get_sandbox_status()},
                structured_data={"reason": exc.error_type, "command": command, "cwd": self.cwd},
            )

        try:
            if run_in_background:
                snapshot = self.process_manager.start_background(
                    execution_command,
                    cwd=self.cwd,
                    env=execution_env,
                    use_shell=use_shell,
                )
                display_text, truncated = _format_background_snapshot(snapshot, max_output_chars=self.max_output_chars)
                return ToolResult.success(
                    display_text,
                    structured_data={
                        **_snapshot_to_dict(snapshot),
                        "description": description,
                        "truncated": truncated,
                        "sandbox": sandbox_metadata,
                    },
                    metadata={
                        "task_id": snapshot.task_id,
                        "status": snapshot.status,
                        "cwd": snapshot.cwd,
                        "description": description,
                        "truncated": truncated,
                        "sandbox": sandbox_metadata,
                    },
                )

            result = self.process_manager.run(
                execution_command,
                cwd=self.cwd,
                env=execution_env,
                timeout_ms=timeout_ms,
                use_shell=use_shell,
            )
        except (PathResolutionError, FilesystemAccessError, ValueError) as exc:
            return self._tool_error(
                f"命令执行失败: {exc}",
                error_type="invalid_path",
                metadata={"command": command, "cwd": self.cwd},
                structured_data={"reason": "invalid_path", "command": command, "cwd": self.cwd},
            )
        except Exception as exc:
            return self._tool_error(
                f"命令执行失败: {exc}",
                error_type="command_failed",
                metadata={"command": command, "cwd": self.cwd},
                structured_data={"reason": "command_failed", "command": command, "cwd": self.cwd},
            )

        if (
            sandbox_decision.applied
            and result.return_code != 0
            and result.stderr.lstrip().startswith("bwrap:")
        ):
            return self._tool_error(
                f"Sandbox failed to launch: {result.stderr.strip()}",
                error_type="sandbox_launch_failed",
                metadata={"command": command, "cwd": self.cwd, "sandbox": sandbox_metadata},
                structured_data={
                    "reason": "sandbox_launch_failed",
                    "command": command,
                    "cwd": self.cwd,
                    "return_code": result.return_code,
                    "stderr": result.stderr,
                    "sandbox": sandbox_metadata,
                },
            )

        sections = [
            f"命令: {command}",
            f"cwd: {result.cwd}",
            f"退出码: {result.return_code}",
        ]
        if result.timed_out:
            sections.append("状态: timed_out")
        if result.stdout:
            sections.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            sections.append(f"stderr:\n{result.stderr}")
        display_text, truncated = _clip_text("\n\n".join(sections), max_chars=self.max_output_chars)
        return ToolResult.success(
            display_text,
            structured_data={
                "command": command,
                "cwd": result.cwd,
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "description": description,
                "truncated": truncated,
                "sandbox": sandbox_metadata,
            },
            metadata={
                "command": command,
                "cwd": result.cwd,
                "return_code": result.return_code,
                "timed_out": result.timed_out,
                "description": description,
                "truncated": truncated,
                "sandbox": sandbox_metadata,
            },
        )


def register_bash_tool(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    shell: str = "bash",
    command_timeout_ms: int = 120000,
    max_background_tasks: int = 8,
    max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
    process_manager: Optional[ProcessManager] = None,
    sandbox: BashSandboxConfig | None = None,
    expose_in_deferred: bool | None = True,
) -> BashTool:
    tool = BashTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=process_manager,
        sandbox=sandbox,
    )
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_shell_tools(
    registry: ToolRegistry,
    workspace_root: Optional[str] = None,
    *,
    allowed_roots: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    shell: str = "bash",
    command_timeout_ms: int = 120000,
    max_background_tasks: int = 8,
    max_output_chars: int = DEFAULT_BASH_OUTPUT_CHARS,
    process_manager: Optional[ProcessManager] = None,
    sandbox: BashSandboxConfig | None = None,
    expose_in_deferred: bool | None = True,
):
    from .task_output import TaskOutputTool
    from .task_stop import TaskStopTool

    manager = process_manager or ProcessManager(shell=shell, max_background_tasks=max_background_tasks)
    bash_tool = register_bash_tool(
        registry,
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=manager,
        sandbox=sandbox,
        expose_in_deferred=expose_in_deferred,
    )
    task_output_tool = TaskOutputTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=manager,
    )
    task_stop_tool = TaskStopTool(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        cwd=cwd,
        shell=shell,
        command_timeout_ms=command_timeout_ms,
        max_background_tasks=max_background_tasks,
        max_output_chars=max_output_chars,
        process_manager=manager,
    )
    registry.register_tool(task_output_tool, expose_in_deferred=expose_in_deferred)
    registry.register_tool(task_stop_tool, expose_in_deferred=expose_in_deferred)
    return bash_tool, task_output_tool, task_stop_tool


__all__ = [
    "BashTool",
    "DEFAULT_BASH_OUTPUT_CHARS",
    "_ShellToolBase",
    "_clip_text",
    "_snapshot_to_dict",
    "_format_background_snapshot",
    "register_bash_tool",
    "register_shell_tools",
]
