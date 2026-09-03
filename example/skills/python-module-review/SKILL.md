---
name: python-module-review
description: Review Python module boundaries, state ownership, and lifecycle behavior.
when_to_use: Use after inspecting Python files under the EasyAgent skill package.
argument-hint: A Python file or module that has already been inspected.
paths: "skill/*.py"
allowed-tools: FileRead, Glob, Grep
---
# Python Module Review

Review `$ARGUMENTS` using the source evidence already gathered from the workspace.

## Workflow

1. Identify the module's public responsibility and its mutable state.
2. Trace creation, binding, execution, cleanup, and persistence boundaries.
3. Check whether optional modules remain disabled until explicitly installed.
4. Verify errors preserve enough structured information for the calling model.
5. Report concrete defects first, then testing gaps. Do not invent behavior not shown by source.

Resolve Skill-local references relative to `${SKILL_DIR}`.
