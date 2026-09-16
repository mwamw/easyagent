from pydantic import BaseModel

from core.permissions.base import BasePermissionEngine
from core.permissions.default import RuleBasedPermissionEngine
from core.permissions.models import (
    PermissionBehavior,
    PermissionDecision,
    PermissionGrant,
    PermissionRequest,
)
from Tool.BaseTool import Tool
from Tool.ToolRegistry import ToolRegistry


class EchoParams(BaseModel):
    text: str


class EchoTool(Tool):
    def __init__(self):
        super().__init__("Echo", "Echo text", EchoParams, read_only=True)

    def run(self, parameters):
        return parameters["text"]


def test_registry_builds_a_permission_request_without_running_or_authorizing():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())

    request = registry.build_permission_request("Echo", {"text": "hello"})

    assert isinstance(request, PermissionRequest)
    assert request.tool_name == "Echo"
    assert request.arguments == {"text": "hello"}
    assert request.descriptor.tool_name == "Echo"


def test_default_engine_returns_the_three_decision_kinds():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    engine = RuleBasedPermissionEngine()
    request = registry.build_permission_request("Echo", {"text": "hello"})

    assert engine.authorize(request).behavior == PermissionBehavior.ALLOW
    engine.add_rule(tool_name="Echo", behavior=PermissionBehavior.ASK)
    assert engine.authorize(request).behavior == PermissionBehavior.ASK
    engine.add_rule(tool_name="Echo", behavior=PermissionBehavior.DENY)
    assert engine.authorize(request).behavior == PermissionBehavior.DENY


def test_custom_permission_engine_only_needs_authorize_request():
    class DenyAll(BasePermissionEngine):
        def authorize(self, request):
            return PermissionDecision(PermissionBehavior.DENY, "custom policy")

    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    request = registry.build_permission_request("Echo", {"text": "hello"})
    decision = registry.authorize_request(request, permission_engine=DenyAll())
    assert decision.behavior == PermissionBehavior.DENY
    assert decision.reason == "custom policy"


def test_request_grant_is_consumed_and_explicit_deny_still_wins():
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    request = registry.build_permission_request("Echo", {"text": "hello"})
    engine = RuleBasedPermissionEngine(load_policies=False)
    engine.add_rule(tool_name="Echo", behavior="ask")
    engine.grant(PermissionGrant(tool_name="Echo", arguments={"text": "hello"}))

    assert engine.authorize(request).behavior == PermissionBehavior.ALLOW
    assert engine.authorize(request).behavior == PermissionBehavior.ASK

    engine.grant(PermissionGrant(tool_name="Echo", arguments={"text": "hello"}))
    engine.add_rule(tool_name="Echo", behavior="deny")
    assert engine.authorize(request).behavior == PermissionBehavior.DENY
