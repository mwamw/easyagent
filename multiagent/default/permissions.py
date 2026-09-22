"""Child authorization assembly, owned by the default multi-agent harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from permissions import BasePermissionEngine, PermissionDecision, PermissionRequest
from permissions.default import RuleBasedPermissionEngine
from runtime import AgentRuntime

ChildPermissionFactory = Callable[[BasePermissionEngine, AgentRuntime], BasePermissionEngine]


class InheritedPermissionEngine(BasePermissionEngine):
    """Borrow authorization logic without rebinding the parent's engine.

    Custom implementations receive the child's PermissionRequest unchanged.
    A stateful custom policy should instead be assembled by a permission factory;
    this view does not isolate arbitrary state hidden inside a custom engine.
    """

    def __init__(self, parent: BasePermissionEngine) -> None:
        self.parent = parent

    def authorize(self, request: PermissionRequest) -> PermissionDecision:
        return self.parent.authorize(request)

    async def aauthorize(self, request: PermissionRequest) -> PermissionDecision:
        return await self.parent.aauthorize(request)


def inherit_permission_engine(
    parent: BasePermissionEngine, runtime: AgentRuntime,
) -> BasePermissionEngine:
    """Snapshot built-in policy data; preserve custom policies by delegation.

    Do not reconstruct arbitrary subclasses: their constructors and extra
    authorization semantics are not part of the permission contract.
    """
    if type(parent) is not RuleBasedPermissionEngine:
        return InheritedPermissionEngine(parent)
    state = deepcopy(parent.export_state())
    state["approval_mode"] = runtime.permission_mode
    state["grants"] = []
    child = RuleBasedPermissionEngine(
        workspace_root=runtime.workspace_root, load_policies=False,
    )
    child.restore_state(state)
    return child
