"""
FileManagerSkill — 本地文件管理技能

封装本地文件读取、搜索与精确编辑工具。
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool

logger = logging.getLogger(__name__)


class FileManagerSkill(BaseSkill):
    """
    文件管理技能

    适合在工作区内做源码阅读、模式检索、精确替换和整文件写入。
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        allowed_roots: Optional[Iterable[str]] = None,
        cwd: Optional[str] = None,
    ):
        config = SkillConfig(
            name="file_manager",
            description="本地文件管理技能，支持读取、搜索、精确编辑和整文件写入。",
            listing_description="Read/search/edit workspace files safely.",
            when_to_use="当你需要定位、读取、修改工作区文件，且希望以最小 diff 精确操作时使用。",
            version="1.0.0",
            tags=["filesystem", "workspace", "edit", "search"],
            priority=6,
        )
        super().__init__(config)
        self.workspace_root = workspace_root
        self.allowed_roots = tuple(allowed_roots) if allowed_roots is not None else None
        self.cwd = cwd

    def get_tools(self) -> List["Tool"]:
        """返回文件读写与搜索工具。"""
        from Tool.builtin import FileEditTool, FileReadTool, FileWriteTool, GlobTool, GrepTool, ListTool

        return [
            FileReadTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
            ),
            ListTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
            ),
            GlobTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
            ),
            GrepTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
            ),
            FileEditTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
            ),
            FileWriteTool(
                workspace_root=self.workspace_root,
                allowed_roots=self.allowed_roots,
                cwd=self.cwd,
            ),
        ]

    def get_prompt(self) -> str:
        """返回文件管理使用指南。"""
        return """## 文件管理能力
你具备本地文件管理能力，适合在工作区内做安全、精确的读写操作。

推荐工作流：
- 先用 `List` 查看目录骨架和当前层级结构，避免直接用 shell `ls -al`
- 先用 `Glob` / `Grep` 缩小候选范围，再对具体文件使用 `FileRead`
- 只改局部文本时优先用 `FileEdit`，并确保 `old_string` 真实存在
- 需要新建文件或整体重写文件时再使用 `FileWrite`
- 修改已有文件前，先读取当前内容，避免基于过期上下文改错位置

行为要求：
- 尽量做最小必要改动，不要无依据地整文件重写
- 先理解上下文，再编辑
- 修改后应继续读取、搜索或运行验证，确认变更范围正确
"""
