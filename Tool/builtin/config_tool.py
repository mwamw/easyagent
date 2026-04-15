"""Claude-style Config tool for reading/updating runtime config."""

from __future__ import annotations

import os
from typing import Any, Optional, get_args, get_origin

from pydantic import TypeAdapter

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeConfigInput
from core.Config import Config


CONFIG_PROMPT = """用于读取或更新当前 Agent 的运行时配置。
- `value` 为空时读取当前值。
- 仅修改内存中的 live Config，不会自动写回环境变量或配置文件。"""


def _split_roots(value: str) -> list[str]:
    if os.pathsep in value:
        parts = value.split(os.pathsep)
    else:
        parts = value.split(",")
    return [item.strip() for item in parts if item.strip()]


class ConfigTool(Tool):
    def __init__(self, *, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        super().__init__(
            name="Config",
            description="读取或更新当前 Agent 的运行时配置。",
            parameters=ClaudeConfigInput,
            guidance="value 为空时读取；value 非空时更新当前 live Config。主要用于调试或切换少量运行参数。",
            prompt=CONFIG_PROMPT,
            read_only=False,
            destructive=False,
            supports_parallel=False,
            source="builtin",
            tags=["config", "claude_code"],
        )

    def _coerce_value(self, setting: str, value: Any) -> Any:
        fields = self.config.__class__.model_fields
        field = fields.get(setting)
        if field is None:
            raise KeyError(setting)
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in {list, tuple} and isinstance(value, str):
            value = _split_roots(value)
        elif origin is None and annotation in {list, tuple} and isinstance(value, str):
            value = _split_roots(value)
        elif origin is None and annotation is bool and isinstance(value, str):
            value = value.strip()
        elif origin is not None and list in args and isinstance(value, str):
            value = _split_roots(value)
        adapter = TypeAdapter(annotation)
        return adapter.validate_python(value)

    def run(self, parameters: dict) -> ToolResult:
        setting = str(parameters.get("setting", "")).strip()
        value = parameters.get("value")
        fields = self.config.__class__.model_fields

        if not setting:
            return ToolResult.error("错误：setting 不能为空。", error_type="invalid_parameters")
        if setting not in fields:
            return ToolResult.error(
                f"错误：未知配置项: {setting}",
                error_type="unknown_setting",
                metadata={"setting": setting, "known_settings": sorted(fields.keys())},
            )

        current_value = getattr(self.config, setting)
        if value is None:
            return ToolResult.success(
                f"{setting} = {current_value!r}",
                structured_data={"setting": setting, "value": current_value},
                metadata={"setting": setting, "value": current_value},
            )

        try:
            coerced_value = self._coerce_value(setting, value)
        except Exception as exc:
            return ToolResult.error(
                f"更新配置失败: {exc}",
                error_type="invalid_value",
                metadata={"setting": setting, "value": value},
            )

        setattr(self.config, setting, coerced_value)
        return ToolResult.success(
            f"已更新配置 {setting}",
            structured_data={
                "setting": setting,
                "previousValue": current_value,
                "newValue": getattr(self.config, setting),
            },
            metadata={
                "setting": setting,
                "previous_value": current_value,
                "new_value": getattr(self.config, setting),
            },
        )


def register_config_tool(registry: ToolRegistry, *, config: Optional[Config] = None) -> ConfigTool:
    tool = ConfigTool(config=config)
    registry.register_tool(tool)
    return tool


__all__ = ["ConfigTool", "register_config_tool"]
