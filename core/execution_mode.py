"""Execution mode state for plan/execute workflows."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"


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

    def enter_plan_mode(self, *, allowed_actions: list[str] | None = None) -> PlanModeState:
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

    def exit_plan_mode(self, *, allowed_actions: list[str] | None = None) -> PlanModeState:
        self.state.mode = ExecutionMode.EXECUTE
        self.state.exit_requested = False
        if allowed_actions is not None:
            self.state.allowed_actions = list(allowed_actions)
        return self.state

    def export_state(self) -> dict[str, Any]:
        return self.state.model_dump(mode="python")

    def restore_state(self, state: dict[str, Any] | None) -> None:
        self.state = PlanModeState.model_validate(state or {})

