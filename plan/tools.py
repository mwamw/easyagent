"""Plan-mode interaction tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.claude_compat.models import ClaudeExitPlanModeInput

if TYPE_CHECKING:
    from .base import BasePlanMode


ENTER_PLAN_MODE_PROMPT = """Use this tool when you are about to start a non-trivial implementation task. Getting user sign-off on your approach before writing code prevents wasted effort and ensures alignment.

When to use:
- New features, multiple valid approaches, architectural decisions, or multi-file changes.
- Unclear requirements where exploration is needed before choosing an implementation.

When not to use:
- Simple, obvious fixes or tasks with specific, detailed instructions.
- Pure research or exploration that does not require implementation planning.

What happens in plan mode:
1. Explore the codebase using read-only tools.
2. Understand existing patterns and architecture.
3. Design an implementation approach.
4. Present the plan to the user.
5. Use AskUserQuestion to clarify requirements.
6. Use ExitPlanMode to request approval before implementation.

This tool interrupts execution and returns a mode-change request to the caller. It does not itself grant permission or create a plan file."""

EXIT_PLAN_MODE_PROMPT = """Use this tool when you are in plan mode, have finished preparing your plan, and are ready for user approval.

When to use:
- Only when planning implementation steps for a task that requires code changes.
- Do not use it for a research-only task.

Before using:
- Resolve questions about requirements or approach with AskUserQuestion.
- Present a complete, unambiguous plan in the conversation.
- Use ExitPlanMode to request approval, not AskUserQuestion.

This tool interrupts execution and returns allowedPrompts to the caller for permission handling. It does not read a plan file or automatically approve the plan."""


class EnterPlanModeInput(BaseModel):
    reason: str = Field(default="", description="Reason for entering plan mode")
    allowedActions: list[str] = Field(default_factory=list, description="Action categories allowed during planning")


def _interrupt_result(
    *,
    message: str,
    error_type: str,
    structured_data: dict,
    metadata: dict,
) -> ToolResult:
    return ToolResult(
        status="needs_confirmation",
        content=message,
        display_text=message,
        structured_data=structured_data,
        metadata=metadata,
        error_type=error_type,
    )


class EnterPlanModeTool(Tool):
    def __init__(self, *, plan_manager: "BasePlanMode | None" = None):
        super().__init__(
            name="EnterPlanMode",
            description="Requests plan mode and interrupts execution for caller approval.",
            parameters=EnterPlanModeInput,
            guidance="Use when requirements or implementation choices need investigation.",
            prompt=ENTER_PLAN_MODE_PROMPT,
            read_only=True,
            destructive=False,
            supports_parallel=False,
            source="builtin",
            tags=["plan", "interaction", "claude_code"],
        )
        self.plan_manager = plan_manager

    def run(self, parameters: dict) -> ToolResult:
        allowed_actions = list(parameters.get("allowedActions") or [])
        reason = str(parameters.get("reason", "")).strip()
        message = "请求进入 plan 模式，等待调用方确认。"
        payload = {
            "allowedActions": allowed_actions,
            "reason": reason,
            "message": message,
        }
        return _interrupt_result(
            message=message,
            error_type="enter_plan_mode_requested",
            structured_data=payload,
            metadata={"interaction_type": "enter_plan_mode", **payload},
        )


class ExitPlanModeTool(Tool):
    def __init__(self, *, plan_manager: "BasePlanMode | None" = None):
        super().__init__(
            name="ExitPlanMode",
            description="Requests plan approval and returns requested permissions to the caller.",
            parameters=ClaudeExitPlanModeInput,
            guidance="Use when the plan is ready for approval before implementation.",
            prompt=EXIT_PLAN_MODE_PROMPT,
            read_only=True,
            destructive=False,
            supports_parallel=False,
            source="builtin",
            tags=["plan", "claude_code"],
        )
        self.plan_manager = plan_manager

    def run(self, parameters: dict) -> ToolResult:
        allowed_prompts = list(parameters.get("allowedPrompts") or [])
        if self.plan_manager is not None and self.plan_manager.is_active:
            self.plan_manager.request_exit(
                allowed_actions=[
                    str(item.get("tool") or item.get("prompt") or "").strip()
                    for item in allowed_prompts
                    if isinstance(item, dict)
                    and str(item.get("tool") or item.get("prompt") or "").strip()
                ]
            )
        message = "请求退出 plan 模式，等待调用方确认允许的执行权限。"
        payload = {
            "allowedPrompts": allowed_prompts,
            "message": message,
        }
        return _interrupt_result(
            message=message,
            error_type="exit_plan_mode_requested",
            structured_data=payload,
            metadata={"interaction_type": "exit_plan_mode", **payload},
        )


def register_enter_plan_mode_tool(
    registry: ToolRegistry,
    *,
    expose_in_deferred: bool | None = True,
    plan_manager: "BasePlanMode | None" = None,
) -> EnterPlanModeTool:
    tool = EnterPlanModeTool(plan_manager=plan_manager)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_exit_plan_mode_tool(
    registry: ToolRegistry,
    *,
    expose_in_deferred: bool | None = True,
    plan_manager: "BasePlanMode | None" = None,
) -> ExitPlanModeTool:
    tool = ExitPlanModeTool(plan_manager=plan_manager)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = [
    "EnterPlanModeInput",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "register_enter_plan_mode_tool",
    "register_exit_plan_mode_tool",
]
