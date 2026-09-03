"""Low-frequency interaction tools that interrupt the caller/UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeAskUserQuestionInput


ASK_USER_PROMPT = """用于结构化向用户提问。
- 当信息缺失且继续执行风险较高时使用。
- 该工具不会直接得到答案，而是会中断当前 tool loop，把问题交回调用方/UI。"""

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
            description="向用户发起结构化提问，并中断当前执行等待回答。",
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


def register_ask_user_question_tool(
    registry: ToolRegistry,
    *,
    expose_in_deferred: bool | None = True,
) -> AskUserQuestionTool:
    tool = AskUserQuestionTool()
    registry.register_tool(tool, expose_in_deferred=expose_in_deferred)
    return tool


__all__ = [
    "AskUserQuestionTool",
    "register_ask_user_question_tool",
]
