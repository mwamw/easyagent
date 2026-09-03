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


SKILL_TOOL_PROMPT = """Load one available Agent Skill and activate its instructions.

Use this tool when:
- A Skill listed in `<available_skills>` matches the user's task.
- The task needs a specialized workflow, domain policy, or reusable procedure from that Skill.
- You need the Skill's full instructions before deciding which steps or tools to use.

How it works:
- Pass the exact Skill name from `<available_skills>` in `skill`.
- Put any user-supplied operands, target names, paths, issue IDs, or other command arguments in `args`.
- For an inline Skill, the full `SKILL.md` body is injected as a user-role meta message after this tool result. Continue the task using those instructions on the next model turn.
- The injected instructions remain available only until the current Agent `invoke` finishes. A later `invoke` must call `skill_tool` again.
- A Skill may temporarily expose deferred tools or add narrowly scoped allow rules declared by `allowed-tools`; normal explicit deny rules still take precedence.
- A Skill declared with `context: fork` runs synchronously in a subagent and returns that subagent's result, agent ID, and output file.

Important constraints:
- Never invent a Skill name. Use only names shown in `<available_skills>`.
- Do not claim that a Skill has been applied merely because it was listed; call this tool first.
- If the conversation already contains a `<skill name="...">` block for that Skill, follow the loaded instructions instead of invoking it recursively.
- Do not repeatedly load the same Skill with the same arguments in one invocation. The tool reports `alreadyActive` when it is already loaded.
- Skills with `disable-model-invocation: true` cannot be called by the model and are omitted from the available listing.
- A successful inline result is not the final user answer. It means the instructions are ready for the next reasoning step.
"""


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
