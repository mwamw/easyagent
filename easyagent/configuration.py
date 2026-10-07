"""Public configuration-loading entry points; existing Agent APIs are unchanged."""
from configuration import (
    AgentBuilder, AgentConfiguration, BuildContext, ConfigurationError,
    ConfigurationPaths, ConfigurationRegistry, Options, default_registry,
    load_agent, read_configuration, read_configuration_source, validate_agent,
)

__all__ = [
    "AgentBuilder", "AgentConfiguration", "BuildContext", "ConfigurationError",
    "ConfigurationPaths", "ConfigurationRegistry", "Options", "default_registry",
    "load_agent", "read_configuration", "read_configuration_source", "validate_agent",
]

if __name__ == "__main__":
    from configuration.__main__ import main
    raise SystemExit(main())
