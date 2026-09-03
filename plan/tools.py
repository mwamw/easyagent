"""Plan-mode interaction tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool, ToolResult
from Tool.ToolRegistry import ToolRegistry
from Tool.claude_compat.models import ClaudeExitPlanModeInput

if TYPE_CHECKING:
    from .manager import PlanModeManager


ENTER_PLAN_MODE_PROMPT = """用于请求进入 plan 模式。
- 适合在继续执行存在较大不确定性时，先进入规划阶段。
- 该工具会中断当前调用，把模式切换请求交回调用方/UI。"""

EXIT_PLAN_MODE_PROMPT = """用于请求退出 plan 模式并声明后续允许的操作类别。
- 它本身不执行权限切换，而是把请求结构化抛给调用方/UI。"""


class EnterPlanModeInput(BaseModel):
    reason: str = Field(default="", description="进入 plan 模式的原因")
    allowedActions: list[str] = Field(default_factory=list, description="计划阶段允许的动作类别")


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
    def __init__(self, *, plan_manager: "PlanModeManager | None" = None):
        super().__init__(
            name="EnterPlanMode",
            description="请求进入 plan 模式，并中断当前执行等待调用方切换模式。",
            parameters=EnterPlanModeInput,
            guidance="适合在需求尚不明确或需要先做方案分析时使用。",
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
    def __init__(self, *, plan_manager: "PlanModeManager | None" = None):
        super().__init__(
            name="ExitPlanMode",
            description="请求退出 plan 模式，并把允许的执行权限交回调用方处理。",
            parameters=ClaudeExitPlanModeInput,
            guidance="适合在计划已确认、准备进入执行阶段时使用。调用后会中断当前 tool loop。",
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
    plan_manager: "PlanModeManager | None" = None,
) -> EnterPlanModeTool:
    tool = EnterPlanModeTool(plan_manager=plan_manager)
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


def register_exit_plan_mode_tool(
    registry: ToolRegistry,
    *,
    expose_in_deferred: bool | None = True,
    plan_manager: "PlanModeManager | None" = None,
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
