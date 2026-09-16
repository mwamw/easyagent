# Linux Bash Sandbox

EasyAgent can run `BashTool` commands through Linux Bubblewrap. This is a
configurable Bash execution backend, not an Agent module: Executor, Context and
other tools do not depend on it.

```python
from easyagent.tools import (
    BashSandboxConfig,
    BashSandboxMode,
    SandboxNetworkMode,
    ToolRegistry,
    register_shell_tools,
)

registry = ToolRegistry()
register_shell_tools(
    registry,
    workspace_root=".",
    sandbox=BashSandboxConfig(
        enabled=True,
        mode=BashSandboxMode.REGULAR,
        allow_unsandboxed_commands=False,
        fail_if_unavailable=True,
        network=SandboxNetworkMode.DENY,
        deny_read=(),
        deny_write=(),
    ),
)
```

## Modes

- `enabled=False`: execute through the normal local `ProcessManager`.
- `REGULAR`: sandbox Bash while retaining normal permission decisions.
- `AUTO_ALLOW`: explicit permission rules still win; otherwise a request that
  will actually enter the sandbox may be auto-allowed by the default engine.
- `allow_unsandboxed_commands=False`: strict override policy. Missing
  dependencies, excluded commands and `dangerouslyDisableSandbox` fail closed.

`fail_if_unavailable=True` rejects tool registration if the configured sandbox
cannot start. When fallback is allowed, results explicitly report why execution
was unsandboxed; fallback is never silent.

## Linux enforcement

The backend starts with a read-only `/`, rebinds the workspace and configured
roots as writable, creates a private `/dev` and `/proc`, and isolates the PID
namespace. `network="deny"` also creates a network namespace; `"inherit"`
retains host networking. Foreground and background commands use the same
sandbox plan.

Read-denied directories are hidden by `tmpfs`; read-denied files are masked.
Write-denied paths inside writable roots are rebound read-only. Linux path
globs are rejected instead of silently weakening policy. Configured paths must
already exist.

## Security boundary

Permission controls whether an action is authorized; Bubblewrap restricts what
an authorized Bash process and its children can access. This implementation
does not sandbox FileRead/FileEdit, MCP, WebFetch or other tools. Bubblewrap is
not a VM and shares the host kernel.

The first Linux implementation supports complete network denial or inherited
networking. Domain allowlists require the separate HTTP/SOCKS proxy bridge used
by Claude Code and are not claimed here.
