"""Data model for directory-based Agent Skills."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


SkillExecutionContext = Literal["inline", "fork"]
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class SkillManifest(BaseModel):
    """Metadata indexed from one ``SKILL.md`` frontmatter block.

    The instruction body is intentionally not stored here. It is read only when
    ``skill_tool`` invokes the Skill, which keeps normal requests lightweight.
    """

    name: str
    description: str
    when_to_use: str = ""
    directory: str
    file_path: str
    allowed_tools: list[str] = Field(default_factory=list)
    argument_hint: str = ""
    context: SkillExecutionContext = "inline"
    agent: str | None = None
    model: str | None = None
    paths: list[str] = Field(default_factory=list)
    disable_model_invocation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Skill name must be non-empty")
        if _SKILL_NAME_PATTERN.fullmatch(normalized) is None:
            raise ValueError(
                "Skill name must match ^[a-z0-9][a-z0-9-]{0,63}$"
            )
        return normalized

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Skill description must be non-empty")
        return normalized


__all__ = ["SkillExecutionContext", "SkillManifest"]
