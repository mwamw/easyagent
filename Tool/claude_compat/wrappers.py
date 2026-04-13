"""Helpers for exposing EasyAgent tools with Claude Code compatible names."""

from __future__ import annotations

from typing import Any, Optional, Type

from pydantic import BaseModel

from ..BaseTool import Tool
from .catalog import get_claude_tool_definition


class ClaudeCompatTool(Tool):
    """Base Tool carrying Claude Code compatibility metadata."""

    def __init__(
        self,
        *,
        claude_name: str,
        description: Optional[str] = None,
        parameters: Optional[Type[BaseModel]] = None,
        guidance: str = "",
        read_only: Optional[bool] = None,
        destructive: Optional[bool] = None,
        requires_confirmation: bool = False,
        supports_parallel: bool = True,
        output_mode: str = "text",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        definition = get_claude_tool_definition(claude_name)
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("compat_layer", "claude_code")
        merged_metadata.setdefault("claude_tool_name", claude_name)
        super().__init__(
            name=claude_name,
            description=description or definition.description,
            parameters=parameters or definition.parameters_model,
            guidance=guidance,
            read_only=definition.read_only if read_only is None else read_only,
            destructive=definition.destructive if destructive is None else destructive,
            requires_confirmation=requires_confirmation,
            supports_parallel=supports_parallel,
            output_mode=output_mode,  # type: ignore[arg-type]
            source="claude_compat",
            tags=list(tags or definition.tags),
            metadata=merged_metadata,
        )


class ClaudeCompatDelegatingTool(ClaudeCompatTool):
    """Expose an existing EasyAgent tool under a Claude compatible name."""

    def __init__(
        self,
        *,
        claude_name: str,
        delegate: Tool,
        description: Optional[str] = None,
        parameters: Optional[Type[BaseModel]] = None,
        guidance: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.delegate = delegate
        spec = delegate.get_spec()
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("delegate_tool_name", delegate.name)
        super().__init__(
            claude_name=claude_name,
            description=description or spec.description,
            parameters=parameters or spec.parameters_model,
            guidance=guidance or spec.guidance,
            read_only=spec.read_only,
            destructive=spec.destructive,
            requires_confirmation=spec.requires_confirmation,
            supports_parallel=spec.supports_parallel,
            output_mode=spec.output_mode,
            tags=["claude_compat", *spec.tags],
            metadata=merged_metadata,
        )

    def run(self, parameters: dict) -> Any:
        return self.delegate.run(parameters)

    async def arun(self, parameters: dict) -> Any:
        return await self.delegate.arun(parameters)
