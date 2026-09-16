"""Permission primitives for EasyAgent."""

from .base import BasePermissionEngine
from .models import (
    ApprovalMode,
    AuthorizationContext,
    PermissionBehavior,
    PermissionDecision,
    PermissionRequest,
    PermissionGrant,
    ToolPermissionDescriptor,
)
from .default import RuleBasedPermissionEngine
from .default.policy import PolicyRule, PolicySet
from .grants import PermissionGrantLedger

PermissionEngine = RuleBasedPermissionEngine

__all__ = [
    "PermissionBehavior",
    "PermissionDecision",
    "PermissionEngine",
    "ApprovalMode",
    "AuthorizationContext",
    "PermissionRequest",
    "PermissionGrant",
    "ToolPermissionDescriptor",
    "BasePermissionEngine",
    "RuleBasedPermissionEngine",
    "PolicyRule",
    "PolicySet",
    "PermissionGrantLedger",
]
