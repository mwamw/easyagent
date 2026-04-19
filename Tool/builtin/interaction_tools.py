"""Low-frequency interaction tools that interrupt the caller/UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeAskUserQuestionInput, ClaudeExitPlanModeInput


ASK_USER_PROMPT = """用于结构化向用户提问。
- 当信息缺失且继续执行风险较高时使用。
- 该工具不会直接得到答案，而是会中断当前 tool loop，把问题交回调用方/UI。"""

EXIT_PLAN_MODE_PROMPT = """用于请求退出 plan 模式并声明后续允许的操作类别。
- 它本身不执行权限切换，而是把请求结构化抛给调用方/UI。"""

ENTER_PLAN_MODE_PROMPT = """用于请求进入 plan 模式。
- 适合在继续执行存在较大不确定性时，先进入规划阶段。
- 该工具会中断当前调用，把模式切换请求交回调用方/UI。"""


class EnterPlanModeInput(BaseModel):
    reason: str = Field(default="", description="进入 plan 模式的原因")
    allowedActions: list[str] = Field(default_factory=list, description="计划阶段允许的动作类别")


def _interrupt_result(
    *,
    message: str,
    error_type: str,
    structured_data: dict[str, Any],
    metadata: dict[str, Any],
) -> ToolResult:
    return ToolResult(
        status="needs_confirmation",
        content=message,
        display_text=message,
        structured_data=structured_data,
        metadata=metadata,
        error_type=error_type,
    )


class AskUserQuestionTool(Tool):
    def __init__(self):
        super().__init__(
            name="AskUserQuestion",
            description="结构化向用户提问，并中断当前执行等待外部回答。",
            parameters=ClaudeAskUserQuestionInput,
            guidance="适合在存在 2-4 个清晰选项时请求用户决策。调用后会中断当前 tool loop。",
            prompt=ASK_USER_PROMPT,
            read_only=True,
            destructive=False,
            supports_parallel=False,
            source="builtin",
            tags=["interaction", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        questions = list(parameters.get("questions") or [])
        source = parameters.get("source")
        question_count = len(questions)
        message = f"需要用户回答 {question_count} 个结构化问题后才能继续执行。"
        payload = {
            "questions": questions,
            "source": source,
            "message": message,
        }
        return _interrupt_result(
            message=message,
            error_type="ask_user_question",
            structured_data=payload,
            metadata={"interaction_type": "ask_user_question", **payload},
        )


class EnterPlanModeTool(Tool):
    def __init__(self):
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
    def __init__(self):
        super().__init__(
            name="ExitPlanMode",
            description="请求退出 plan 模式，并把允许的权限类别交回调用方/UI 处理。",
            parameters=ClaudeExitPlanModeInput,
            guidance="适合在计划已确认、准备进入执行阶段时使用。调用后会中断当前 tool loop。",
            prompt=EXIT_PLAN_MODE_PROMPT,
            read_only=True,
            destructive=False,
            supports_parallel=False,
            source="builtin",
            tags=["plan", "claude_code"],
        )

    def run(self, parameters: dict) -> ToolResult:
        allowed_prompts = list(parameters.get("allowedPrompts") or [])
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


def register_ask_user_question_tool(registry: ToolRegistry) -> AskUserQuestionTool:
    tool = AskUserQuestionTool()
    registry.register_tool(tool)
    return tool


def register_enter_plan_mode_tool(registry: ToolRegistry) -> EnterPlanModeTool:
    tool = EnterPlanModeTool()
    registry.register_tool(tool)
    return tool


def register_exit_plan_mode_tool(registry: ToolRegistry) -> ExitPlanModeTool:
    tool = ExitPlanModeTool()
    registry.register_tool(tool)
    return tool


__all__ = [
    "AskUserQuestionTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "register_ask_user_question_tool",
    "register_enter_plan_mode_tool",
    "register_exit_plan_mode_tool",
]
