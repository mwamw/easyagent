"""
LinuxOpsSkill — 本地 Shell / Linux 运维技能

封装 Bash 工具与命令执行策略。
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool

logger = logging.getLogger(__name__)


class LinuxOpsSkill(BaseSkill):
    """
    Linux 运维技能

    适合运行测试、构建、格式化、git 查询、日志排查与脚本执行。
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
        shell: str = "bash",
        command_timeout_ms: int = 120000,
        max_background_tasks: int = 8,
        max_output_chars: int = 120000,
    ):
        config = SkillConfig(
            name="linux_ops",
            description="本地 shell 与 Linux 运维技能，支持命令执行、后台任务和真实环境验证。",
            listing_description="Run shell commands for tests, builds, logs, and git inspection.",
            when_to_use="当你需要通过 shell 在真实工作区里运行测试、构建、脚本、git 查询或排查日志时使用。",
            version="1.0.0",
            tags=["shell", "linux", "ops", "bash", "verification"],
            priority=5,
        )
        super().__init__(config)
        self.workspace_root = workspace_root
        self.allowed_roots = tuple(allowed_roots) if allowed_roots is not None else None
        self.cwd = cwd
        self.shell = shell
        self.command_timeout_ms = command_timeout_ms
        self.max_background_tasks = max_background_tasks
        self.max_output_chars = max_output_chars

    def get_tools(self) -> List["Tool"]:
        """返回 Bash 工具。"""
        from Tool.builtin import BashTool

        return [
            BashTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
                shell=self.shell,
                command_timeout_ms=self.command_timeout_ms,
                max_background_tasks=self.max_background_tasks,
                max_output_chars=self.max_output_chars,
            )
        ]

    def get_prompt(self) -> str:
        """返回 shell 运维使用指南。"""
        return """## Shell / Linux 运维能力
你具备在当前工作区执行本地 shell 命令的能力。

适用场景：
- 运行测试、构建、lint、格式化
- 查询 git 状态、diff、分支信息
- 排查日志、检查目录结构、确认生成产物
- 启动需要持续观察输出的后台任务

工具约定：
- 通过 `Bash` 执行本地命令
- 长任务使用 `Bash(run_in_background=true)` 后继续观察输出

执行原则：
- 先明确目标，再执行最小必要命令
- 长时间任务请显式使用后台模式，并主动轮询输出
- 不要把文件编辑职责塞给 shell；精确改文件优先使用文件工具
- 涉及高副作用命令时，要清楚说明边界并核对结果
"""
