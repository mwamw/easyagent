"""
YAML / Markdown Skill 加载器

支持从 YAML 或 Markdown 文件加载 Skill 定义，实现零代码定义 Skill。

YAML 格式示例::

    name: web_researcher
    description: "Web 研究技能"
    version: "1.0"
    tags: [research, web]
    priority: 5
    tools:
      - builtin: web_search
      - builtin: calculator
    prompt: |
      ## Web 研究能力
      当用户询问实时信息时，使用搜索工具。
    config:
      max_results: 10

Markdown 格式示例::

    ---
    name: code_helper
    description: 代码辅助技能
    version: "1.0"
    tags: [code, programming]
    priority: 3
    tools:
      - builtin: calculator
    ---

    ## 代码辅助能力

    你具备代码分析和辅助编程的能力。

    - 当用户提供代码段时，进行分析和优化建议
    - 使用 calculator 进行数学计算
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillConfig

logger = logging.getLogger(__name__)

# ==================== 内置工具映射 ====================

_BUILTIN_TOOL_FACTORIES: Dict[str, Any] = {}


def _get_builtin_tool(name: str, **kwargs):
    """延迟导入并创建内置工具实例"""
    # 延迟初始化映射
    if not _BUILTIN_TOOL_FACTORIES:
        _BUILTIN_TOOL_FACTORIES.update({
            "web_search": lambda **kw: _create_web_search(**kw),
            "calculator": lambda **kw: _create_calculator(**kw),
        })

    factory = _BUILTIN_TOOL_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"未知的内置工具: '{name}'。可用: {list(_BUILTIN_TOOL_FACTORIES.keys())}")
    return factory(**kwargs)


def _create_web_search(**kwargs):
    from Tool.builtin.search import WebSearchTool
    return WebSearchTool(**kwargs)


def _create_calculator(**kwargs):
    from Tool.builtin.calculator import CalculatorTool
    return CalculatorTool(**kwargs)


# ==================== YAML Skill ====================


class YAMLSkill(BaseSkill):
    """
    从 YAML 配置创建的 Skill

    将 YAML 中声明的 tools 和 prompt 映射为 BaseSkill 接口实现。
    """

    def __init__(
        self,
        config: SkillConfig,
        tool_defs: List[Dict[str, Any]],
        prompt_text: str = "",
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self._tool_defs = tool_defs
        self._prompt_text = prompt_text
        self._extra_config = extra_config or {}

    def get_tools(self) -> list:
        """根据 YAML 中的 tools 定义创建工具实例"""
        tools = []
        for tool_def in self._tool_defs:
            try:
                if "builtin" in tool_def:
                    builtin_name = tool_def["builtin"]
                    tool_kwargs = {k: v for k, v in tool_def.items() if k != "builtin"}
                    tool = _get_builtin_tool(builtin_name, **tool_kwargs)
                    tools.append(tool)
                else:
                    logger.warning("不支持的工具定义格式: %s", tool_def)
            except Exception as e:
                logger.error("创建工具失败: %s，错误: %s", tool_def, e)
        return tools

    def get_prompt(self) -> str:
        """返回 YAML 中定义的 prompt"""
        return self._prompt_text


class YAMLSkillLoader:
    """YAML Skill 加载器"""

    @staticmethod
    def load(yaml_path: str) -> YAMLSkill:
        """
        从单个 YAML 文件加载 Skill

        Args:
            yaml_path: YAML 文件路径

        Returns:
            YAMLSkill 实例

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: YAML 格式不正确
        """
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML Skill 文件不存在: {yaml_path}")

        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装 PyYAML：pip install pyyaml")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML 文件顶层必须是字典: {yaml_path}")

        return YAMLSkillLoader._parse_data(data)

    @staticmethod
    def load_directory(dir_path: str) -> List[YAMLSkill]:
        """
        从目录加载所有 YAML Skill

        Args:
            dir_path: 目录路径

        Returns:
            YAMLSkill 实例列表
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        skills = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith((".yaml", ".yml")):
                filepath = os.path.join(dir_path, filename)
                try:
                    skill = YAMLSkillLoader.load(filepath)
                    skills.append(skill)
                except Exception as e:
                    logger.warning("加载 YAML Skill '%s' 失败: %s", filepath, e)

        return skills

    @staticmethod
    def _parse_data(data: Dict[str, Any]) -> YAMLSkill:
        """解析 YAML 数据为 YAMLSkill"""
        name = data.get("name")
        if not name:
            raise ValueError("YAML Skill 必须包含 'name' 字段")

        config = SkillConfig(
            name=name,
            description=data.get("description", ""),
            version=str(data.get("version", "1.0.0")),
            tags=data.get("tags", []),
            priority=int(data.get("priority", 0)),
            auto_activate=data.get("auto_activate", True),
            dependencies=data.get("dependencies", []),
            extra=data.get("config", {}),
        )

        # 解析工具定义
        tool_defs = data.get("tools", [])
        # 标准化：将 "- builtin: name" 简写转为 {"builtin": "name"}
        normalized_tools = []
        for td in tool_defs:
            if isinstance(td, str):
                normalized_tools.append({"builtin": td})
            elif isinstance(td, dict):
                normalized_tools.append(td)
            else:
                logger.warning("忽略不支持的工具定义: %s", td)
        tool_defs = normalized_tools

        prompt_text = data.get("prompt", "")

        return YAMLSkill(
            config=config,
            tool_defs=tool_defs,
            prompt_text=prompt_text,
            extra_config=data.get("config"),
        )


# ==================== Markdown Skill ====================


class MarkdownSkill(BaseSkill):
    """
    从 Markdown 文件创建的 Skill

    Markdown 格式：YAML frontmatter + Markdown body。
    frontmatter 中定义配置和工具，Markdown body 作为 prompt。
    """

    def __init__(
        self,
        config: SkillConfig,
        tool_defs: List[Dict[str, Any]],
        prompt_text: str = "",
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self._tool_defs = tool_defs
        self._prompt_text = prompt_text
        self._extra_config = extra_config or {}

    def get_tools(self) -> list:
        """根据 frontmatter 中的 tools 定义创建工具实例"""
        tools = []
        for tool_def in self._tool_defs:
            try:
                if "builtin" in tool_def:
                    builtin_name = tool_def["builtin"]
                    tool_kwargs = {k: v for k, v in tool_def.items() if k != "builtin"}
                    tool = _get_builtin_tool(builtin_name, **tool_kwargs)
                    tools.append(tool)
            except Exception as e:
                logger.error("创建工具失败: %s，错误: %s", tool_def, e)
        return tools

    def get_prompt(self) -> str:
        """返回 Markdown body 作为 prompt"""
        return self._prompt_text


class MarkdownSkillLoader:
    """Markdown Skill 加载器"""

    # YAML frontmatter 正则：匹配 --- ... --- 之间的内容
    _FRONTMATTER_RE = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n",
        re.DOTALL,
    )

    @staticmethod
    def load(md_path: str) -> MarkdownSkill:
        """
        从单个 Markdown 文件加载 Skill

        Markdown 格式要求：
        1. 文件以 YAML frontmatter (--- ... ---) 开头
        2. frontmatter 包含 name 等配置
        3. frontmatter 之后的 Markdown 内容作为 prompt

        Args:
            md_path: Markdown 文件路径

        Returns:
            MarkdownSkill 实例

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 格式不正确
        """
        if not os.path.isfile(md_path):
            raise FileNotFoundError(f"Markdown Skill 文件不存在: {md_path}")

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        return MarkdownSkillLoader._parse_content(content, md_path)

    @staticmethod
    def load_directory(dir_path: str) -> List[MarkdownSkill]:
        """
        从目录加载所有 Markdown Skill

        Args:
            dir_path: 目录路径

        Returns:
            MarkdownSkill 实例列表
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        skills = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith(".md"):
                filepath = os.path.join(dir_path, filename)
                try:
                    skill = MarkdownSkillLoader.load(filepath)
                    skills.append(skill)
                except Exception as e:
                    logger.warning("加载 Markdown Skill '%s' 失败: %s", filepath, e)

        return skills

    @staticmethod
    def _parse_content(content: str, source_path: str = "") -> MarkdownSkill:
        """解析 Markdown 内容"""
        match = MarkdownSkillLoader._FRONTMATTER_RE.match(content)

        if not match:
            raise ValueError(
                f"Markdown Skill 文件必须以 YAML frontmatter (--- ... ---) 开头: {source_path}"
            )

        frontmatter_text = match.group(1)
        body_text = content[match.end():].strip()

        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装 PyYAML：pip install pyyaml")

        data = yaml.safe_load(frontmatter_text)
        if not isinstance(data, dict):
            raise ValueError(f"Markdown frontmatter 必须是字典格式: {source_path}")

        name = data.get("name")
        if not name:
            # 尝试从文件名推断
            if source_path:
                name = os.path.splitext(os.path.basename(source_path))[0]
            else:
                raise ValueError("Markdown Skill 必须在 frontmatter 中指定 'name'")

        config = SkillConfig(
            name=name,
            description=data.get("description", ""),
            version=str(data.get("version", "1.0.0")),
            tags=data.get("tags", []),
            priority=int(data.get("priority", 0)),
            auto_activate=data.get("auto_activate", True),
            dependencies=data.get("dependencies", []),
            extra=data.get("config", {}),
        )

        # 解析工具定义
        tool_defs = data.get("tools", [])
        normalized_tools = []
        for td in tool_defs:
            if isinstance(td, str):
                normalized_tools.append({"builtin": td})
            elif isinstance(td, dict):
                normalized_tools.append(td)
        tool_defs = normalized_tools

        return MarkdownSkill(
            config=config,
            tool_defs=tool_defs,
            prompt_text=body_text,
            extra_config=data.get("config"),
        )
