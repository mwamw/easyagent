"""Structured Bash sandbox failures."""


class BashSandboxError(RuntimeError):
    error_type = "sandbox_error"


class SandboxUnavailableError(BashSandboxError):
    error_type = "sandbox_unavailable"


class SandboxConfigurationError(BashSandboxError):
    error_type = "sandbox_configuration_error"


class SandboxPathError(BashSandboxError):
    error_type = "sandbox_path_error"


class SandboxBypassDeniedError(BashSandboxError):
    error_type = "sandbox_bypass_denied"


class SandboxLaunchError(BashSandboxError):
    error_type = "sandbox_launch_failed"
