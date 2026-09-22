"""Configuration models owned by the bundled rule-based permission engine."""

from enum import Enum


class ApprovalMode(str, Enum):
    """High-level policy presets implemented by the default engine."""

    ASK_FOR_APPROVAL = "ask_for_approval"
    AUTO_APPROVE_SAFE = "auto_approve_safe"
    FULL_ACCESS = "full_access"
