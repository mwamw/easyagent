"""Errors safe to display without including configuration values or secrets."""

class ConfigurationError(ValueError):
    def __init__(self, location: str, message: str):
        self.location = location
        self.message = message
        super().__init__(f"{location}: {message}")
