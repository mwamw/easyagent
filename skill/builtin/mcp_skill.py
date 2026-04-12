"""MCP skills for EasyAgent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from skill.base import BaseSkill, SkillConfig

if TYPE_CHECKING:
    from Tool.BaseTool import Tool
    from Tool.builtin.mcp_tool import MCPToolManager
    from skill.registry import SkillRegistry

logger = logging.getLogger(__name__)


def _normalize_name(source: str) -> str:
    name = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = name.rsplit(".", 1)[0]
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_") or "unknown"


def _format_prompt_arguments(arguments: List[Dict[str, Any]]) -> str:
    if not arguments:
        return ""

    lines = ["### Prompt 参数"]
    for item in arguments:
        name = item.get("name", "")
        required = "必填" if item.get("required") else "可选"
        description = item.get("description", "")
        line = f"- `{name}` ({required})"
        if description:
            line += f": {description}"
        lines.append(line)
    return "\n".join(lines)


class MCPPromptSkill(BaseSkill):
    """Wrap one MCP prompt as an on-demand inline Skill."""

    def __init__(
        self,
        manager: "MCPToolManager",
        prompt_name: str,
        prompt_info: Dict[str, Any],
        skill_name: str,
        prompt_arguments: Optional[Dict[str, str]] = None,
    ):
        description = str(prompt_info.get("description", "") or f"MCP prompt: {prompt_name}")
        arguments = list(prompt_info.get("arguments", []) or [])
        config = SkillConfig(
            name=skill_name,
            description=description,
            listing_description=description,
            when_to_use=description or f"当需要 MCP prompt `{prompt_name}` 的远程指令模板时",
            version="1.0.0",
            tags=["mcp", "prompt", manager.server_label],
            priority=5,
            auto_activate=False,
            exposure_mode="on_demand",
            execution_mode="inline",
            source_type="mcp_prompt",
            source_path=f"mcp://{manager.server_label}/prompts/{prompt_name}",
            extra={
                "mcp_server": manager.server_label,
                "mcp_prompt_name": prompt_name,
                "mcp_prompt_arguments": arguments,
            },
        )
        super().__init__(config)
        self._manager = manager
        self._prompt_name = prompt_name
        self._prompt_info = dict(prompt_info)
        self._prompt_arguments = dict(prompt_arguments or {})

    def get_tools(self) -> List["Tool"]:
        return []

    def set_prompt_arguments(self, prompt_arguments: Dict[str, Any]) -> None:
        self._prompt_arguments = {
            str(key): str(value)
            for key, value in dict(prompt_arguments).items()
        }

    def get_prompt(self) -> str:
        return self.get_body_prompt()

    def get_body_prompt(self) -> str:
        arguments = list(self._prompt_info.get("arguments", []) or [])
        argument_block = _format_prompt_arguments(arguments)

        try:
            messages = self._manager.get_remote_prompt(self._prompt_name, self._prompt_arguments)
        except Exception as exc:
            lines = [
                f"## MCP Prompt: {self._prompt_name}",
                f"该 Skill 来自 MCP 服务 `{self._manager.server_label}` 的远程 prompt。",
            ]
            if argument_block:
                lines.append(argument_block)
            lines.append(
                f"当前未能直接拉取远程 prompt 正文：{exc}。"
                "如该 prompt 需要参数，请先准备完整参数后再重新调用相关 Skill。"
            )
            return "\n\n".join(lines)

        lines = [
            f"## MCP Prompt: {self._prompt_name}",
            f"该 Skill 来自 MCP 服务 `{self._manager.server_label}` 的远程 prompt。",
        ]
        if argument_block:
            lines.append(argument_block)
        if self._prompt_arguments:
            supplied = ", ".join(f"{key}={value}" for key, value in self._prompt_arguments.items())
            lines.append(f"### 当前注入参数\n- {supplied}")
        lines.append("### 远程 Prompt 消息")

        if not messages:
            lines.append("（该 prompt 当前未返回任何消息正文）")
            return "\n\n".join(lines)

        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            lines.append(f"#### {role}\n{content or '（空内容）'}")

        return "\n\n".join(lines)


class MCPSkill(BaseSkill):
    """Attach MCP tools/resources and optionally register MCP prompts as Skills."""

    def __init__(
        self,
        server_source: Any,
        server_args: Optional[List[str]] = None,
        transport_type: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        tool_prefix: str = "",
        auto_connect: bool = True,
        skill_name: Optional[str] = None,
        *,
        include_resource_tools: bool = True,
        resource_tool_prefix: Optional[str] = None,
        prompt_registry: Optional["SkillRegistry"] = None,
        register_prompt_skills: bool = True,
        prompt_skill_prefix: Optional[str] = None,
        **transport_kwargs: Any,
    ):
        normalized_name = _normalize_name(str(server_source))
        name = skill_name or f"mcp_{normalized_name}"
        config = SkillConfig(
            name=name,
            description=f"MCP 远程服务: {server_source}",
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
        self._include_resource_tools = include_resource_tools
        self._resource_tool_prefix = resource_tool_prefix
        self._prompt_registry = prompt_registry
        self._register_prompt_skills = register_prompt_skills
        self._prompt_skill_prefix = prompt_skill_prefix or f"{name}_"
        self._manager: Optional["MCPToolManager"] = None
        self._registered_prompt_skills: List[str] = []

    def _ensure_manager(self) -> "MCPToolManager":
        from Tool.builtin.mcp_tool import MCPToolManager

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
        return self._manager

    def get_tools(self) -> List["Tool"]:
        from Tool.builtin.mcp_tool import MCPWrappedTool

        manager = self._ensure_manager()
        try:
            remote_tools = manager.list_remote_tools()
            tools: List["Tool"] = [
                MCPWrappedTool(manager, info, prefix=self._tool_prefix)
                for info in remote_tools
            ]
            if self._include_resource_tools:
                tools.extend(manager.build_resource_tools(self._resource_tool_prefix))
            return tools
        except Exception as exc:
            logger.error("获取 MCP 远程能力失败: %s", exc)
            return []

    def get_prompt(self) -> str:
        lines = [
            "## MCP 远程能力",
            f"你拥有来自 MCP 服务（{self._server_source}）的远程能力。",
            "- 远程调用可能有延迟，避免无意义重复调用。",
            "- 严格遵守当前 tools 集合中实际暴露的 MCP 工具名称和参数 schema。",
        ]
        if self._include_resource_tools:
            lines.append("- 你还可以通过资源工具读取 MCP 暴露的只读资源。")
        if self._prompt_registry is not None and self._register_prompt_skills:
            lines.append("- 该服务的 MCP prompts 会注册为 on-demand Skills，按需使用 `skill_tool` 调用。")
        return "\n".join(lines)

    def on_activate(self, agent: Any) -> None:
        if self._prompt_registry is None or not self._register_prompt_skills:
            return

        try:
            manager = self._ensure_manager()
            self._registered_prompt_skills = manager.register_prompt_skills(
                self._prompt_registry,
                skill_prefix=self._prompt_skill_prefix,
            )
        except Exception as exc:
            logger.warning("注册 MCP prompt skills 失败: %s", exc)

    def on_deactivate(self, agent: Any) -> None:
        if self._prompt_registry is not None:
            for name in list(self._registered_prompt_skills):
                try:
                    self._prompt_registry.unregister(name)
                except KeyError:
                    pass
                except Exception as exc:
                    logger.warning("移除 MCP prompt skill '%s' 失败: %s", name, exc)
            self._registered_prompt_skills.clear()

        if self._manager is not None:
            try:
                self._manager.close()
            except Exception as exc:
                logger.warning("关闭 MCP 连接失败: %s", exc)
            self._manager = None
