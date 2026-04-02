"""
MCPSkill — MCP 远程工具技能

将 MCP 服务器上的远程工具封装为 Skill，支持动态发现和调用。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool

logger = logging.getLogger(__name__)


class MCPSkill(BaseSkill):
    """
    MCP 远程工具技能

    连接 MCP 服务器，动态发现远程工具并注册到 Agent。

    Example::

        skill = MCPSkill(
            server_source="path/to/mcp_server.py",
            transport_type="stdio",
        )
        agent.with_skill(skill)
    """

    def __init__(
        self,
        server_source: Any,
        server_args: Optional[List[str]] = None,
        transport_type: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        tool_prefix: str = "",
        auto_connect: bool = True,
        skill_name: Optional[str] = None,
        **transport_kwargs: Any,
    ):
        """
        初始化 MCPSkill

        Args:
            server_source: MCP 服务器源标识
            server_args: 启动服务器的命令行参数
            transport_type: 传输类型
            env: 环境变量
            tool_prefix: 工具名前缀
            auto_connect: 是否自动连接
            skill_name: Skill 名称（默认基于 server_source 生成）
            **transport_kwargs: 传输层额外参数
        """
        name = skill_name or f"mcp_{self._normalize_name(str(server_source))}"
        config = SkillConfig(
            name=name,
            description=f"MCP 远程工具服务: {server_source}",
            version="1.0.0",
            tags=["mcp", "remote", "tools"],
            priority=4,
        )
        super().__init__(config)

        self._server_source = server_source
        self._server_args = server_args
        self._transport_type = transport_type
        self._env = env
        self._tool_prefix = tool_prefix
        self._auto_connect = auto_connect
        self._transport_kwargs = transport_kwargs
        self._manager = None

    def get_tools(self) -> List["Tool"]:
        """动态获取 MCP 远程工具并包装"""
        from Tool.builtin.mcp_tool import MCPToolManager, MCPWrappedTool

        if self._manager is None:
            self._manager = MCPToolManager(
                server_source=self._server_source,
                server_args=self._server_args,
                transport_type=self._transport_type,
                env=self._env,
                tool_prefix=self._tool_prefix,
                auto_connect=self._auto_connect,
                **self._transport_kwargs,
            )

        try:
            remote_tools = self._manager.list_remote_tools()
            return [
                MCPWrappedTool(self._manager, info, prefix=self._tool_prefix)
                for info in remote_tools
            ]
        except Exception as e:
            logger.error("获取 MCP 远程工具失败: %s", e)
            return []

    def get_prompt(self) -> str:
        """返回 MCP 工具使用指南"""
        return f"""## MCP 远程工具
你拥有来自 MCP 服务（{self._server_source}）的远程工具能力。
这些工具通过网络调用远程服务执行，请注意：
- 远程调用可能有延迟，避免不必要的调用
- 确保传递正确的参数格式
"""

    def on_deactivate(self, agent: Any) -> None:
        """停用时关闭 MCP 连接"""
        if self._manager is not None:
            try:
                self._manager.close()
                logger.info("MCP 连接已关闭")
            except Exception as e:
                logger.warning("关闭 MCP 连接失败: %s", e)
            self._manager = None

    @staticmethod
    def _normalize_name(source: str) -> str:
        """标准化名称"""
        # 取文件名或最后一段路径
        name = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        name = name.rsplit(".", 1)[0]
        # 清理非法字符
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_") or "unknown"
