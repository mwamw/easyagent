from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent import BasicAgent
from core.history import coerce_canonical_message
from core.llm import EasyLLM
from core.permissions import PermissionBehavior, PermissionRule
from runtime import RuntimeEventType
from skill import SkillManager, SkillTool, discover_skill_files, load_skill_body, load_skill_manifest
from Tool import Tool
from Tool.BaseTool import ToolResult
from Tool.builtin.agent_tool import AgentTool
from Tool.runtime import SubagentRequest


class StubLLM(EasyLLM):
    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.base_url = "http://mock.local/v1"
        self.api_key = "mock-key"
        self.max_tokens = 256
        self.client = None


class SkillFlowProvider:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def build_request(
        self,
        messages,
        *,
        system_prompt=None,
        tools=None,
        temperature=None,
        reasoning=None,
        stream=False,
        **kwargs,
    ) -> dict:
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(list(messages))
        return {
            "messages": request_messages,
            "tools": tools,
            "temperature": temperature,
            "reasoning": reasoning,
            "stream": stream,
            **kwargs,
        }

    def apply_cache_policy(self, request, request_input):
        return request

    def invoke_raw(self, request):
        self.requests.append(request)
        has_skill_result = any(
            message.get("role") == "tool" and message.get("name") == "skill_tool"
            for message in request["messages"]
            if isinstance(message, dict)
        )
        if not has_skill_result:
            return SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call-skill",
                        function=SimpleNamespace(
                            name="skill_tool",
                            arguments='{"skill":"review","args":"src/app.py"}',
                        ),
                    )
                ],
                usage=None,
            )
        return SimpleNamespace(
            content="review complete",
            reasoning_content=None,
            tool_calls=[],
            usage=None,
        )


class SkillFlowLLM(EasyLLM):
    def __init__(self) -> None:
        self.provider_name = "openai"
        self.model = "test-model"
        self.base_url = "http://test.local/v1"
        self.api_key = "test-key"
        self.max_tokens = 256
        self.temperature = 0.2
        self.timeout = 60
        self.kwargs = {}
        self._provider = SkillFlowProvider()
        self.client = None


class EmptyInput(BaseModel):
    pass


class EchoTool(Tool):
    def __init__(self):
        super().__init__("echo_tool", "Echo", EmptyInput, read_only=True)

    def run(self, parameters: dict) -> str:
        return "echo"


class PathInput(BaseModel):
    file_path: str


class FilesystemProbeTool(Tool):
    def __init__(self):
        super().__init__(
            "FilesystemProbe",
            "Report a touched file",
            PathInput,
            read_only=True,
            tags=["filesystem"],
        )

    def run(self, parameters: dict) -> ToolResult:
        return ToolResult.success(
            "path touched",
            structured_data={"file_path": parameters["file_path"]},
        )


class ForkAgentInput(BaseModel):
    description: str
    prompt: str
    run_in_background: bool = False
    subagent_type: str | None = None
    model: str | None = None


class ForkAgentTool(Tool):
    def __init__(self):
        super().__init__("Agent", "Run a subagent", ForkAgentInput)
        self.last_parameters: dict = {}

    def run(self, parameters: dict) -> ToolResult:
        self.last_parameters = dict(parameters)
        return ToolResult.success(
            "fork result",
            structured_data={
                "agentId": "agent-child",
                "outputFile": "/tmp/agent-child.md",
                "content": "fork result",
            },
        )


def write_skill(
    parent: Path,
    name: str,
    *,
    description: str = "A focused test workflow",
    extra_frontmatter: str = "",
    body: str = "Follow this exact workflow for $ARGUMENTS in ${SKILL_DIR}.",
) -> Path:
    directory = parent / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                extra_frontmatter.strip(),
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return directory


def publish(agent: BasicAgent, event_type: RuntimeEventType, invocation_id: str = "invoke-test") -> None:
    agent.event_bus.publish(
        event_type,
        agent_id=agent.name,
        invocation_id=invocation_id,
        data={"query": "test"} if event_type == RuntimeEventType.AGENT_INVOKE_STARTED else {},
    )


def test_manifest_is_indexed_without_loading_body(tmp_path: Path):
    directory = write_skill(tmp_path, "review", body="SECRET SKILL BODY")
    manifest = load_skill_manifest(directory / "SKILL.md")

    assert manifest.name == "review"
    assert "SECRET SKILL BODY" not in manifest.model_dump_json()
    assert load_skill_body(manifest, "src/app.py") == "SECRET SKILL BODY"


def test_allowed_tools_parser_preserves_parenthesized_rules(tmp_path: Path):
    directory = write_skill(
        tmp_path,
        "git-review",
        extra_frontmatter="allowed-tools: Bash(git status:*) FileRead, Grep",
    )
    manifest = load_skill_manifest(directory / "SKILL.md")
    assert manifest.allowed_tools == ["Bash(git status:*)", "FileRead", "Grep"]


def test_unsupported_allowed_tool_matcher_fails_during_indexing(tmp_path: Path):
    write_skill(tmp_path, "unsafe", extra_frontmatter="allowed-tools: Custom(scope:*)")

    with pytest.raises(ValueError, match="Unsupported allowed-tools matcher"):
        SkillManager().add_directories([tmp_path])


def test_allowed_tool_prefixes_are_normalized_without_widening_access():
    bash = SkillManager._permission_rule("Bash(git status:*)")
    path = SkillManager._permission_rule("FileRead(src/**)")
    domain = SkillManager._permission_rule("WebFetch(domain:*.example.com)")

    assert bash.matcher == {"command_prefixes": ["git status"]}
    assert path.matcher == {"path_prefixes": ["src/"]}
    assert domain.matcher == {"hosts": ["example.com"]}


def test_inline_skill_rejects_fork_only_agent_and_model_fields(tmp_path: Path):
    write_skill(tmp_path, "invalid", extra_frontmatter="model: specialist-model")

    with pytest.raises(ValueError, match="only with context: fork"):
        SkillManager().add_directories([tmp_path])


def test_paths_parser_supports_comma_lists_braces_and_match_all(tmp_path: Path):
    conditional = write_skill(
        tmp_path,
        "conditional",
        extra_frontmatter='paths: "src/*.{py,pyi}, tests/**"',
    )
    match_all = write_skill(tmp_path, "global", extra_frontmatter="paths: '**'")

    assert load_skill_manifest(conditional / "SKILL.md").paths == [
        "src/*.py",
        "src/*.pyi",
        "tests",
    ]
    assert load_skill_manifest(match_all / "SKILL.md").paths == []


def test_discovery_accepts_single_skill_or_collection(tmp_path: Path):
    first = write_skill(tmp_path, "first")
    write_skill(tmp_path, "second")

    assert discover_skill_files(first) == [first.resolve() / "SKILL.md"]
    assert [path.parent.name for path in discover_skill_files(tmp_path)] == ["first", "second"]


def test_agent_has_no_skill_module_until_with_skill(tmp_path: Path):
    agent = BasicAgent("agent", StubLLM())
    assert agent.skill_manager is None
    assert agent.tool_registry is None

    write_skill(tmp_path, "review")
    returned = agent.with_skill(tmp_path)

    assert returned is agent
    assert isinstance(agent.skill_manager, SkillManager)
    assert agent.tool_registry is not None
    assert agent.tool_registry.has_tool("skill_tool")


def test_with_skill_accepts_multiple_directories_and_chaining(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_c = tmp_path / "c"
    write_skill(root_a, "one")
    write_skill(root_b, "two")
    write_skill(root_c, "three")

    agent = BasicAgent("agent", StubLLM()).with_skill(root_a, root_b).with_skill(root_c)
    assert agent.skill_manager is not None
    assert agent.skill_manager.skill_names == ("one", "three", "two")
    assert agent.tool_registry.get_tool_names().count("skill_tool") == 1


def test_duplicate_names_from_different_roots_are_rejected_transactionally(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_skill(first, "same", body="first")
    write_skill(second, "same", body="second")
    manager = SkillManager().add_directories([first])

    with pytest.raises(ValueError, match="Duplicate Skill name"):
        manager.add_directories([second])

    assert manager.get_skill("same").file_path == str((first / "same" / "SKILL.md").resolve())


def test_listing_is_a_system_reminder_and_does_not_contain_body(tmp_path: Path):
    write_skill(
        tmp_path,
        "review",
        description="Review a code change",
        extra_frontmatter="when_to_use: Use for pull request review",
        body="SECRET BODY",
    )
    agent = BasicAgent("agent", StubLLM()).with_skill(tmp_path)
    template = agent.get_system_prompt_template("review this change")

    assert "SECRET BODY" not in template.render_system()
    reminder = template.render_system_reminders()
    assert "<available_skills>" in reminder
    assert "`review`" in reminder
    assert "Use for pull request review" in reminder


def test_path_conditional_skill_activates_from_filesystem_runtime_event(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_root = tmp_path / "skills"
    write_skill(
        skill_root,
        "python-review",
        extra_frontmatter='paths: "src/**/*.py"',
    )
    agent = (
        BasicAgent("agent", StubLLM())
        .add_tool(FilesystemProbeTool())
        .with_skill(skill_root)
    )
    agent.execution_context.workspace_root = str(workspace)

    assert "`python-review`" not in agent.get_system_prompt_template().render_system_reminders()
    hidden = agent.execute_tool_result("skill_tool", {"skill": "python-review"})
    assert hidden.status == "error"
    assert hidden.error_type == "skill_not_active"
    assert hidden.structured_data["conditionalPaths"] == ["src/**/*.py"]

    agent.event_bus.publish(
        RuntimeEventType.TOOL_INVOKE_COMPLETED,
        agent_id=agent.name,
        invocation_id="invoke-path",
        data={
            "tool_name": "FilesystemProbe",
            "arguments": {"file_path": "src/package/module.py"},
            "result": ToolResult.success(
                "path touched",
                structured_data={
                    "file_path": str(workspace / "src/package/module.py")
                },
            ),
        },
    )

    assert "`python-review`" in agent.get_system_prompt_template().render_system_reminders()
    assert agent.skill_manager.export_state()["activatedPathSkills"] == [
        "python-review"
    ]


def test_skill_tool_queues_full_body_after_result_boundary(tmp_path: Path):
    write_skill(tmp_path, "review", body="Review $ARGUMENTS from ${SKILL_DIR}.")
    agent = BasicAgent("agent", StubLLM()).with_skill(tmp_path)
    publish(agent, RuntimeEventType.AGENT_INVOKE_STARTED)

    result = agent.execute_tool_result("skill_tool", {"skill": "review", "args": "src/app.py"})
    assert result.status == "success"
    assert "Review src/app.py" not in result.to_display_string()
    assert result.structured_data["instructionSource"] in result.to_display_string()
    assert not any("Review src/app.py" in message.text_content() for message in agent.history)

    agent.history_store.append_tool_result(result.to_display_string(), "call-1", "skill_tool")
    agent.metamessage_manager.flush()
    messages = [coerce_canonical_message(item) for item in agent.history]
    assert "Review src/app.py" in messages[-1].text_content()
    assert str((tmp_path / "review").resolve()) in messages[-1].text_content()
    assert messages[-1].metadata["metaMessageLifecycle"] == "invocation"


def test_executor_injects_skill_between_tool_result_and_next_llm_request(tmp_path: Path):
    write_skill(tmp_path, "review", body="Review target $ARGUMENTS with the loaded workflow.")
    llm = SkillFlowLLM()
    agent = BasicAgent("agent", llm).with_skill(tmp_path)

    assert agent.invoke("Review the target") == "review complete"
    assert len(llm.provider.requests) == 2
    second_messages = llm.provider.requests[1]["messages"]
    tool_result_index = next(
        index for index, message in enumerate(second_messages)
        if message.get("role") == "tool" and message.get("name") == "skill_tool"
    )
    skill_index = next(
        index for index, message in enumerate(second_messages)
        if message.get("role") == "user"
        and "Review target src/app.py with the loaded workflow." in str(message.get("content"))
    )
    assert tool_result_index < skill_index
    assert not any(message.metadata.get("source") == "skill" for message in agent.history)


def test_skill_body_and_permissions_are_reclaimed_on_agent_terminal_event(tmp_path: Path):
    write_skill(tmp_path, "review", extra_frontmatter="allowed-tools: echo_tool")
    agent = BasicAgent("agent", StubLLM()).add_tool(EchoTool()).with_skill(tmp_path)
    agent.add_permission_rule(
        PermissionRule(tool_name="echo_tool", behavior=PermissionBehavior.ASK),
        source="session",
    )
    publish(agent, RuntimeEventType.AGENT_INVOKE_STARTED)
    result = agent.execute_tool_result("skill_tool", {"skill": "review"})
    assert result.status == "success"
    assert "skill:review" in agent.permission_context.store.sources
    agent.metamessage_manager.flush()
    assert any(message.metadata.get("source") == "skill" for message in agent.history)

    publish(agent, RuntimeEventType.AGENT_INVOKE_COMPLETED)
    assert "skill:review" not in agent.permission_context.store.sources
    assert not any(message.metadata.get("source") == "skill" for message in agent.history)


def test_same_skill_and_arguments_are_deduplicated_per_invocation(tmp_path: Path):
    write_skill(tmp_path, "review")
    agent = BasicAgent("agent", StubLLM()).with_skill(tmp_path)
    publish(agent, RuntimeEventType.AGENT_INVOKE_STARTED)

    first = agent.execute_tool_result("skill_tool", {"skill": "review", "args": "a"})
    second = agent.execute_tool_result("skill_tool", {"skill": "review", "args": "a"})
    assert first.structured_data["alreadyActive"] is False
    assert second.structured_data["alreadyActive"] is True
    assert len(agent.metamessage_manager.list_pending()) == 1

    publish(agent, RuntimeEventType.AGENT_INVOKE_COMPLETED)
    publish(agent, RuntimeEventType.AGENT_INVOKE_STARTED, "invoke-next")
    third = agent.execute_tool_result("skill_tool", {"skill": "review", "args": "a"})
    assert third.structured_data["alreadyActive"] is False


def test_model_disabled_skill_is_hidden_and_rejected(tmp_path: Path):
    write_skill(tmp_path, "manual", extra_frontmatter="disable-model-invocation: true")
    agent = BasicAgent("agent", StubLLM()).with_skill(tmp_path)

    assert "`manual`" not in agent.get_system_prompt_template().render_system_reminders()
    result = agent.execute_tool_result("skill_tool", {"skill": "manual"})
    assert result.status == "error"
    assert result.error_type == "skill_model_invocation_disabled"


def test_fork_skill_returns_subagent_identity_output_and_result(tmp_path: Path):
    write_skill(
        tmp_path,
        "deep-review",
        extra_frontmatter=(
            "context: fork\n"
            "agent: reviewer\n"
            "model: specialist-model\n"
            "allowed-tools: Agent"
        ),
        body="Review $ARGUMENTS deeply.",
    )
    fork_tool = ForkAgentTool()
    agent = BasicAgent("agent", StubLLM()).add_tool(fork_tool).with_skill(tmp_path)

    result = agent.execute_tool_result(
        "skill_tool",
        {"skill": "deep-review", "args": "src/core.py"},
    )
    assert result.status == "success"
    assert result.structured_data["status"] == "forked"
    assert result.structured_data["agentId"] == "agent-child"
    assert result.structured_data["outputFile"] == "/tmp/agent-child.md"
    assert result.structured_data["result"] == "fork result"
    assert result.structured_data["allowedTools"] == ["Agent"]
    assert "skill:deep-review" not in agent.permission_context.store.sources
    assert fork_tool.last_parameters["subagent_type"] == "reviewer"
    assert fork_tool.last_parameters["model"] == "specialist-model"
    assert "Review src/core.py deeply." in fork_tool.last_parameters["prompt"]
    assert "/tmp/agent-child.md" in result.to_display_string()
    assert "fork result" in result.to_display_string()
    assert agent.metamessage_manager.list_pending() == []


def test_default_subagent_gets_an_agent_local_skill_manager(tmp_path: Path):
    skill_root = tmp_path / "skills"
    write_skill(skill_root, "review")
    parent = BasicAgent("parent", StubLLM()).with_skill(skill_root)
    agent_tool = AgentTool(
        parent_agent=parent,
        storage_dir=str(tmp_path / "agents"),
    )
    child = agent_tool.agent_factory(
        SubagentRequest(
            description="child",
            prompt="inspect the project",
            workspace_root=str(tmp_path),
            allowed_roots=(str(tmp_path),),
        )
    )

    assert child.skill_manager is not None
    assert child.skill_manager is not parent.skill_manager
    child_skill_tool = child.tool_registry.get_tool("skill_tool")
    assert child_skill_tool.manager is child.skill_manager
    assert child_skill_tool.manager is not parent.skill_manager

    child.close()
    agent_tool.agent_runtime.close()
    parent.close()


def test_skill_tool_has_detailed_progressive_disclosure_contract(tmp_path: Path):
    write_skill(tmp_path, "review")
    agent = BasicAgent("agent", StubLLM()).with_skill(tmp_path)
    tool = agent.tool_registry.get_tool("skill_tool")

    assert isinstance(tool, SkillTool)
    description = tool.get_spec().build_schema_description()
    assert "<available_skills>" in description
    assert "current Agent `invoke`" in description
    assert "context: fork" in description
    assert "$ARGUMENTS" in tool.get_spec().parameter_schema()["properties"]["args"]["description"]
