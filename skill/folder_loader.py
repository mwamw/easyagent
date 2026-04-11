"""
Folder-Based Skill 加载器 (Claude Code 风格)

支持从一个包含 README.md (或 skill.md) 和可选 tools.py 脚本的目录作为一个完整的 Skill 加载。
"""
from __future__ import annotations

import logging
import os
import importlib.util
from typing import Any, Dict, List, Optional
import sys

from .base import SkillConfig
from .yaml_loader import MarkdownSkill, MarkdownSkillLoader

logger = logging.getLogger(__name__)


class FolderSkill(MarkdownSkill):
    """
    基于文件夹创建的 Skill

    继承自 MarkdownSkill（因为其核心说明和配置在 md 文件中），
    并扩展支持动态加载文件夹内的 Python 工具实例。
    """

    def __init__(
        self,
        config: SkillConfig,
        tool_defs: List[Dict[str, Any]],
        prompt_text: str = "",
        extra_config: Optional[Dict[str, Any]] = None,
        dynamic_tools: Optional[list] = None,
    ):
        super().__init__(config, tool_defs, prompt_text, extra_config)
        self._dynamic_tools = dynamic_tools or []

    def get_tools(self) -> list:
        """
        返回所有工具，包括 Markdown frontmatter 里定义的 builtin 工具
        以及丛 tools.py 动态加载进来的工具。
        """
        # 获取 Markdown 中定义的工具 (比如 builtin)
        base_tools = super().get_tools()
        
        # 将静态配置的工具和动态加载的工具合并返回
        return base_tools + self._dynamic_tools


class FolderSkillLoader:
    """Folder Skill 加载器"""

    @staticmethod
    def load(dir_path: str) -> FolderSkill:
        """
        从目录加载 Skill。

        目录要求：
        1. 必须包含 skill.md 或 README.md，里面有 YAML frontmatter
        2. 可以包含 tools.py，里面定义工具实例。

        Args:
            dir_path: 目录路径

        Returns:
            FolderSkill 实例

        Raises:
            FileNotFoundError: 目录或必须的被说明文件不存在
            ValueError: 格式不正确
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Folder Skill 目录不存在: {dir_path}")

        # 1. 查找必须的 Markdown 说明文件
        md_file = None
        for filename in ["skill.md", "README.md"]:
            candidate = os.path.join(dir_path, filename)
            if os.path.isfile(candidate):
                md_file = candidate
                break
        
        if not md_file:
            raise FileNotFoundError(f"Folder Skill 目录 {dir_path} 中缺少 skill.md 或 README.md 文件")

        # 使用 MarkdownSkillLoader 解析 frontmatter 和 prompt 内容
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        base_md_skill = MarkdownSkillLoader._parse_content(content, md_file)
        if not base_md_skill.config.source_path:
            base_md_skill.config.source_path = dir_path
        base_md_skill.config.source_type = "folder"
        if not base_md_skill.config.exposure_mode:
            base_md_skill.config.exposure_mode = "on_demand"
        if not base_md_skill.config.execution_mode:
            base_md_skill.config.execution_mode = "inline"
        
        # 2. 检查是否有 tools.py, 如果有则尝试动态加载里面导出的工具
        dynamic_tools = []
        tools_script = os.path.join(dir_path, "tools.py")
        if os.path.isfile(tools_script):
            dynamic_tools = FolderSkillLoader._load_tools_from_python(tools_script)

        return FolderSkill(
            config=base_md_skill.config,
            tool_defs=base_md_skill._tool_defs,
            prompt_text=base_md_skill._prompt_text,
            extra_config=base_md_skill._extra_config,
            dynamic_tools=dynamic_tools
        )

    @staticmethod
    def _load_tools_from_python(filepath: str) -> list:
        """从 Python 文件中动态提取工具实例"""
        from Tool.BaseTool import Tool

        tools = []
        module_name = "folder_skill_dynamic_tools_" + os.path.basename(os.path.dirname(filepath))
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                return tools

            module = importlib.util.module_from_spec(spec)
            
            # 将该文件所在目录临时加入到 sys.path，以便 tools.py 能够导入内部模块
            dir_path = os.path.dirname(filepath)
            sys.path.insert(0, dir_path)
            try:
                spec.loader.exec_module(module)
            finally:
                if sys.path[0] == dir_path:
                    sys.path.pop(0)

            # 1. 检查是否存在 get_tools() 入口函数
            if hasattr(module, "get_tools") and callable(getattr(module, "get_tools")):
                custom_tools = getattr(module, "get_tools")()
                if isinstance(custom_tools, list):
                    for tb in custom_tools:
                        if isinstance(tb, Tool):
                            tools.append(tb)
                        else:
                            logger.warning("get_tools() 返回了非 Tool 实例: %s", type(tb))
                else:
                    logger.warning("get_tools() 应返回列表，但返回了: %s", type(custom_tools))
                return tools

            # 2. 如果没有显式 get_tools()，自动扫描所有的 Tool 子类并实例化
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                # 检查是否是一个直接继承 BaseTool 的子类 (不是抽象基类自身)
                if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                    try:
                        # 尝试无参实例化
                        tool_instance = attr()  # type: ignore
                        tools.append(tool_instance)
                    except Exception as e:
                        logger.warning("自动实例化由 '%s' 定义的 %s 失败，请检查是否需要参数: %s", filepath, attr_name, e)

        except Exception as e:
            logger.error("加载包含动态工具的 py 文件 '%s' 失败: %s", filepath, e)

        return tools
