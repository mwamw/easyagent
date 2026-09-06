"""The model-facing tool for progressive Skill disclosure."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from Tool.BaseTool import Tool, ToolResult

from .manager import (
    SkillManager,
    SkillModelInvocationDisabledError,
    SkillNotActiveError,
)


SKILL_TOOL_PROMPT = """Execute a skill within the current Agent `invoke`.

When users ask you to perform tasks, check if any available skills match. Skills provide specialized capabilities and domain knowledge.

How to invoke:
- Use skill with the name from the <available_skills> listing.
- Supply optional args for the skill's input.
- Available skills are listed in system-reminder messages.
- When a skill matches the user's request, invoke skill_tool before following its workflow.
- Never claim to have used a skill without actually calling this tool.
- Do not invoke a skill that is already active in this invocation.
- Do not use this tool for built-in CLI or TUI commands.

Execution:
- For an inline skill, the full SKILL.md body is injected after the tool result as a <skill name="..."> block. Follow those instructions in the next model step.
- If that block is already present for this invocation, follow it instead of recursively loading the skill.
- Instructions remain active only for the current Agent invocation; load the skill again in a later invocation when needed.
- A skill configured with context: fork delegates its task to Agent instead.
- Use the returned skill directory to resolve supporting scripts, references, and assets. Do not guess instructions that have not been loaded.
- Skill-scoped tool permissions cannot override a denial or expose unavailable tools."""


class SkillToolInput(BaseModel):
    skill: str = Field(
        description="Exact Skill name from the <available_skills> listing, without a leading slash."
    )
    args: str = Field(
        default="",
        description=(
            "Optional raw arguments for the Skill, such as a path, issue ID, target, or user request. "
            "They replace $ARGUMENTS in SKILL.md and are also included in the injected context."
        ),
    )


class SkillTool(Tool):
    def __init__(self, manager: SkillManager) -> None:
        self.manager = manager
        super().__init__(
            name="skill_tool",
            description="Load and execute one indexed Agent Skill by exact name.",
            parameters=SkillToolInput,
            guidance=SKILL_TOOL_PROMPT,
            read_only=True,
            supports_parallel=False,
            source="builtin",
            tags=["skill", "progressive-disclosure"],
            side_effect_level="none",
            resource_scope=["agent_history", "permissions", "tools"],
            expose_in_deferred=True,
        )

    def run(self, parameters: dict) -> ToolResult:
        skill_name = str(parameters.get("skill") or "").strip().lstrip("/")
        args = str(parameters.get("args") or "")
        try:
            payload = self.manager.invoke(skill_name, args=args, model_initiated=True)
        except KeyError as exc:
            return ToolResult.error(
                str(exc),
                error_type="skill_not_found",
                metadata={"skill": skill_name},
            )
        except SkillModelInvocationDisabledError as exc:
            return ToolResult.error(
                str(exc),
                error_type="skill_model_invocation_disabled",
                metadata={"skill": skill_name},
            )
        except SkillNotActiveError as exc:
            paths = list(exc.manifest.paths)
            return ToolResult(
                status="error",
                content=(
                    f"{exc}. Conditional path patterns: "
                    f"{', '.join(paths) or 'none'}. Continue normal file inspection; "
                    "the Skill will appear in <available_skills> after a matching "
                    "filesystem tool succeeds."
                ),
                error_type="skill_not_active",
                structured_data={
                    "skill": skill_name,
                    "conditionalPaths": paths,
                },
                metadata={"skill": skill_name},
            )
        except Exception as exc:
            return ToolResult.error(
                f"Failed to load Skill '{skill_name}': {exc}",
                error_type="skill_invoke_failed",
                metadata={"skill": skill_name},
            )

        if payload["status"] == "forked":
            content = (
                f"Skill `{skill_name}` completed in a forked subagent. "
                f"Agent ID: {payload.get('agentId') or 'unknown'}. "
                f"Output file: {payload.get('outputFile') or 'unavailable'}."
            )
        elif payload["alreadyActive"]:
            content = (
                f"Skill `{skill_name}` is already active with these arguments for the current "
                "Agent invocation. Continue using the previously injected instructions."
            )
        else:
            content = (
                f"Skill `{skill_name}` loaded successfully. Its full instructions will appear "
                "after this tool result on the next model request and remain active until the "
                "current Agent invocation finishes."
            )
        display_text = (
            f"{content}\n\n"
            "Structured Skill result:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        )
        return ToolResult.success(
            content,
            display_text=display_text,
            structured_data=payload,
            metadata={"skill": skill_name, "skill_status": payload["status"]},
        )


__all__ = ["SKILL_TOOL_PROMPT", "SkillTool", "SkillToolInput"]
