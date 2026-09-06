"""Low-frequency interaction tools that interrupt the caller/UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..BaseTool import Tool, ToolResult
from ..ToolRegistry import ToolRegistry
from ..claude_compat.models import ClaudeAskUserQuestionInput


ASK_USER_PROMPT = """Use this tool when you need to ask the user questions during execution. This allows you to:
1. Gather user preferences or requirements.
2. Clarify ambiguous instructions.
3. Get decisions on implementation choices as you work.
4. Offer choices about what direction to take.

Usage notes:
- Provide 1-4 questions, each with 2-4 clearly described options.
- Use multiSelect=true to allow multiple answers for a question.
- If recommending an option, put it first and add "(Recommended)" to its label.
- This tool interrupts the current execution and returns the questions to the caller. Wait for the caller to supply answers.

Plan mode note: Clarify requirements or choose between approaches before finalizing a plan. Use ExitPlanMode for plan approval, not AskUserQuestion."""

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
            description="Asks the user structured questions and interrupts execution to await answers.",
            parameters=ClaudeAskUserQuestionInput,
            guidance="Provide 2-4 clear options per question. The caller handles the resulting interruption.",
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
