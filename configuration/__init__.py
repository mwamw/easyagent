"""Optional declarative assembly on top of the ordinary EasyAgent Python API."""
from .builder import AgentBuilder, load_agent, validate_agent
from .builtins import default_registry
from .errors import ConfigurationError
from .loader import read_configuration, read_configuration_source
from .models import AgentConfiguration, ConfigurationPaths, Options
from .registry import BuildContext, ConfigurationRegistry

__all__ = [
    "AgentBuilder", "AgentConfiguration", "BuildContext", "ConfigurationError",
    "ConfigurationPaths", "ConfigurationRegistry", "Options", "default_registry",
    "load_agent", "read_configuration", "read_configuration_source", "validate_agent",
]
