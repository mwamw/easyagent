"""Plan-mode configuration and runtime state."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_PLAN_ENTER_MESSAGE = (
    "Plan mode is active. Investigate the task, inspect relevant context, and produce or refine "
    "an implementation plan. Do not perform mutating actions, edit files, or execute the plan "
    "until plan mode has been exited. Read-only inspection and planning operations remain allowed."
)

DEFAULT_PLAN_EXIT_MESSAGE = (
    "Plan mode has ended. The earlier plan-only restrictions no longer apply. Continue in execute "
    "mode and use the currently available tools and permissions to carry out the approved work."
)


class ExecutionMode(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"


class PlanModeConfig(BaseModel):
    enter_message: str = Field(default=DEFAULT_PLAN_ENTER_MESSAGE)
    exit_message: str = Field(default=DEFAULT_PLAN_EXIT_MESSAGE)
    allowed_actions: list[str] = Field(default_factory=list)
    register_tools: bool = True


class PlanModeState(BaseModel):
    mode: ExecutionMode = Field(default=ExecutionMode.EXECUTE)
    entered_at: datetime | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    exit_requested: bool = False


class ModeController:
    def __init__(self, state: PlanModeState | None = None):
        self.state = state or PlanModeState()

    @property
    def mode(self) -> ExecutionMode:
        return self.state.mode

    def enter(self, *, allowed_actions: list[str] | None = None) -> PlanModeState:
        self.state.mode = ExecutionMode.PLAN
        self.state.entered_at = datetime.now()
        self.state.allowed_actions = list(allowed_actions or [])
        self.state.exit_requested = False
        return self.state

    def request_exit(self, *, allowed_actions: list[str] | None = None) -> PlanModeState:
        self.state.exit_requested = True
        if allowed_actions is not None:
            self.state.allowed_actions = list(allowed_actions)
        return self.state

    def exit(self) -> PlanModeState:
        self.state.mode = ExecutionMode.EXECUTE
        self.state.exit_requested = False
        return self.state

    def export_state(self) -> dict[str, Any]:
        return self.state.model_dump(mode="python")

    def restore_state(self, state: dict[str, Any] | None) -> None:
        self.state = PlanModeState.model_validate(state or {})


__all__ = [
    "DEFAULT_PLAN_ENTER_MESSAGE",
    "DEFAULT_PLAN_EXIT_MESSAGE",
    "ExecutionMode",
    "ModeController",
    "PlanModeConfig",
    "PlanModeState",
]
