import shutil

import pytest

from permissions.default import RuleBasedPermissionEngine
from permissions.models import PermissionBehavior
from Tool.ToolRegistry import ToolRegistry
from Tool.builtin.bash_sandbox import (
    BashSandboxConfig,
    BashSandboxMode,
    SandboxUnavailableError,
    SandboxNetworkMode,
)
from Tool.builtin.bash_tool import BashTool
from Tool.runtime import ProcessManager


def build_tool(root, config):
    manager = ProcessManager(shell="bash", max_background_tasks=2)
    try:
        tool = BashTool(
            workspace_root=str(root),
            shell="bash",
            process_manager=manager,
            sandbox=config,
        )
    except BaseException:
        manager.close()
        raise
    return tool, manager


def test_disabled_sandbox_runs_normally_and_reports_execution_fact(tmp_path):
    tool, manager = build_tool(tmp_path, BashSandboxConfig())
    try:
        result = tool.run({"command": "printf ok"})
        assert result.structured_data["stdout"] == "ok"
        assert result.structured_data["sandbox"] == {
            "configured": False,
            "applied": False,
            "mode": "regular",
            "backend": "bubblewrap",
            "bypassed": False,
            "fallback": False,
            "reason": "sandbox disabled",
        }
    finally:
        manager.close()


def test_missing_bwrap_falls_back_only_when_policy_allows(tmp_path):
    missing = str(tmp_path / "missing-bwrap")
    tool, manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        bubblewrap_path=missing,
        allow_unsandboxed_commands=True,
    ))
    try:
        result = tool.run({"command": "printf fallback"})
        assert result.structured_data["stdout"] == "fallback"
        assert result.structured_data["sandbox"]["fallback"] is True
        assert result.structured_data["sandbox"]["applied"] is False
    finally:
        manager.close()

    with pytest.raises(SandboxUnavailableError):
        build_tool(tmp_path, BashSandboxConfig(
            enabled=True,
            bubblewrap_path=missing,
            allow_unsandboxed_commands=False,
        ))


def test_network_namespace_unavailability_is_explicit(tmp_path):
    tool, manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.DENY,
        allow_unsandboxed_commands=True,
    ))
    try:
        status = tool.get_sandbox_status()
        result = tool.run({"command": "printf network-policy"})
        if status["available"]:
            assert result.structured_data["sandbox"]["applied"] is True
        else:
            assert status["errors"]
            assert result.structured_data["sandbox"]["fallback"] is True
            assert result.structured_data["sandbox"]["reason"]
    finally:
        manager.close()


def test_strict_sandbox_rejects_dangerous_bypass(tmp_path):
    tool, manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=False,
    ))
    try:
        result = tool.run({
            "command": "printf forbidden",
            "dangerouslyDisableSandbox": True,
        })
        assert result.status == "error"
        assert result.error_type == "sandbox_bypass_denied"
    finally:
        manager.close()


def test_excluded_commands_follow_unsandboxed_fallback_policy(tmp_path):
    open_tool, open_manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        excluded_commands=("printf *",),
        allow_unsandboxed_commands=True,
    ))
    try:
        result = open_tool.run({"command": "printf excluded"})
        assert result.structured_data["stdout"] == "excluded"
        assert result.structured_data["sandbox"]["fallback"] is True
        assert result.structured_data["sandbox"]["applied"] is False
    finally:
        open_manager.close()

    strict_tool, strict_manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        excluded_commands=("printf *",),
        allow_unsandboxed_commands=False,
    ))
    try:
        result = strict_tool.run({"command": "printf excluded"})
        assert result.status == "error"
        assert result.error_type == "sandbox_bypass_denied"
    finally:
        strict_manager.close()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap is not installed")
def test_bubblewrap_allows_workspace_writes_and_blocks_outside_writes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    config = BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=False,
    )
    tool, manager = build_tool(workspace, config)
    try:
        if not tool.get_sandbox_status()["available"]:
            pytest.skip(str(tool.get_sandbox_status()["errors"]))
        inside = tool.run({"command": "printf inside > inside.txt"})
        assert inside.structured_data["return_code"] == 0
        assert (workspace / "inside.txt").read_text() == "inside"

        escaped = tool.run({"command": f"printf escaped > {outside}"})
        assert escaped.structured_data["return_code"] != 0
        assert not outside.exists()
        assert escaped.structured_data["sandbox"]["applied"] is True
    finally:
        manager.close()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap is not installed")
def test_bubblewrap_enforces_read_and_write_denies(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protected = workspace / "protected.txt"
    protected.write_text("original")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret-value")
    config = BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=False,
        deny_write=(str(protected),),
        deny_read=(str(secret),),
    )
    tool, manager = build_tool(workspace, config)
    try:
        if not tool.get_sandbox_status()["available"]:
            pytest.skip(str(tool.get_sandbox_status()["errors"]))
        result = tool.run({
            "command": f"printf changed > {protected}; cat {secret}",
        })
        assert result.structured_data["return_code"] != 0
        assert protected.read_text() == "original"
        assert "secret-value" not in result.structured_data["stdout"]
    finally:
        manager.close()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap is not installed")
def test_background_command_uses_same_sandbox(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "background-outside.txt"
    tool, manager = build_tool(workspace, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=False,
    ))
    try:
        if not tool.get_sandbox_status()["available"]:
            pytest.skip(str(tool.get_sandbox_status()["errors"]))
        started = tool.run({
            "command": f"printf escaped > {outside}",
            "run_in_background": True,
        })
        snapshot = manager.get_task(started.structured_data["task_id"], block=True, timeout_ms=3000)
        assert snapshot.return_code != 0
        assert not outside.exists()
        assert started.structured_data["sandbox"]["applied"] is True
    finally:
        manager.close()


def test_auto_allow_uses_request_specific_sandbox_fact_and_rules_still_win(tmp_path):
    tool, manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        mode=BashSandboxMode.AUTO_ALLOW,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=True,
    ))
    registry = ToolRegistry()
    registry.register_tool(tool)
    engine = RuleBasedPermissionEngine(load_policies=False)
    try:
        request = registry.build_permission_request("Bash", {"command": "printf safe"})
        assert request.extensions["sandbox"]["will_sandbox"] is True
        assert engine.authorize(request).behavior == PermissionBehavior.ALLOW

        bypass = registry.build_permission_request("Bash", {
            "command": "printf host",
            "dangerouslyDisableSandbox": True,
        })
        assert bypass.extensions["sandbox"]["will_sandbox"] is False
        assert engine.authorize(bypass).behavior == PermissionBehavior.ASK

        engine.add_rule(tool_name="Bash", behavior="deny")
        assert engine.authorize(request).behavior == PermissionBehavior.DENY

        ask_engine = RuleBasedPermissionEngine(load_policies=False)
        ask_engine.add_rule(tool_name="Bash", behavior="ask")
        assert ask_engine.authorize(request).behavior == PermissionBehavior.ASK
    finally:
        manager.close()


def test_linux_sandbox_rejects_glob_paths_instead_of_weakening_policy(tmp_path):
    tool, manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=False,
        allow_write=("build/*",),
    ))
    try:
        result = tool.run({"command": "printf no"})
        assert result.status == "error"
        assert result.error_type == "sandbox_path_error"
    finally:
        manager.close()


def test_linux_sandbox_rejects_symlink_policy_paths(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    tool, manager = build_tool(tmp_path, BashSandboxConfig(
        enabled=True,
        network=SandboxNetworkMode.INHERIT,
        allow_unsandboxed_commands=False,
        allow_write=(str(link),),
    ))
    try:
        result = tool.run({"command": "printf no"})
        assert result.status == "error"
        assert result.error_type == "sandbox_path_error"
    finally:
        manager.close()
