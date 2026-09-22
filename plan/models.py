"""Plan-mode configuration and runtime state."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Aligned with utils/messages.ts in Claude Code 2.1.88. EasyAgent has no
# writable plan-file exception or fixed Explore/Plan subagent registry.
DEFAULT_PLAN_ENTER_MESSAGE = """<system-reminder>
Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits, run any non-readonly tools (including changing configs or making commits), or otherwise make changes to the system. Only read-only inspection and planning operations remain allowed until plan mode is exited.

## Plan Workflow

### Phase 1: Initial Understanding
Goal: Gain a comprehensive understanding of the user's request by reading through code and asking questions.
- Focus on the user's request and the associated code. Actively search for existing functions, utilities, and patterns that can be reused -- avoid proposing new code when suitable implementations already exist.
- Explore independent areas with independent read-only tool calls. If delegating research, give each available agent a specific focus and explicit read-only constraints.

### Phase 2: Design
Goal: Design an implementation approach based on the user's intent and exploration results.
- Consider requirements, constraints, and existing architecture.
- When seeking another agent's input, provide filenames, code paths, and relevant findings rather than a vague task title.

### Phase 3: Review
Goal: Review the approach and ensure alignment with the user's intentions.
- Read the critical files identified during exploration.
- Ensure the plan aligns with the original request.
- Use AskUserQuestion, when available, to clarify remaining requirements.

### Phase 4: Final Plan
Goal: Present your final plan in the conversation. This runtime does not provide a writable plan file.
- Begin with Context: explain the problem or need and intended outcome.
- Include only your recommended approach, not all alternatives.
- Keep the plan concise enough to scan, but detailed enough to execute.
- Include paths of critical files to modify and existing functions or utilities to reuse.
- Include a verification section describing how to test the changes end-to-end.

### Phase 5: Request approval
When the plan is complete, use ExitPlanMode, when available, to request approval before implementation. Use AskUserQuestion only to clarify requirements or choose between approaches, not to approve the plan.

At any point, ask for clarification rather than making large assumptions about user intent.
</system-reminder>"""

DEFAULT_PLAN_EXIT_MESSAGE = """<system-reminder>
## Exited Plan Mode

Plan mode has ended. You can now make edits, run tools, and take actions within the currently granted permissions and approved scope. The earlier plan-only restrictions no longer apply.
</system-reminder>"""


class ExecutionMode(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"


class PlanModeConfig(BaseModel):
    enter_message: str = Field(default=DEFAULT_PLAN_ENTER_MESSAGE)
    exit_message: str = Field(default=DEFAULT_PLAN_EXIT_MESSAGE)
    allowed_actions: list[str] = Field(default_factory=list)
    register_tools: bool = True


class PlanModeState(BaseModel):
    entered_at: datetime | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    exit_requested: bool = False


__all__ = [
    "DEFAULT_PLAN_ENTER_MESSAGE",
    "DEFAULT_PLAN_EXIT_MESSAGE",
    "ExecutionMode",
    "PlanModeConfig",
    "PlanModeState",
]
