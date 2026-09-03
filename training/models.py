"""Training export contracts built from observability records."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TrainingDataFormat(str, Enum):
    STEP_SFT = "step_sft"
    TRACE_SFT = "trace_sft"
    AGENTIC_ROLLOUT = "agentic_rollout"


class TrainingExportReport(BaseModel):
    source_records: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    files: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)


__all__ = ["TrainingDataFormat", "TrainingExportReport"]
