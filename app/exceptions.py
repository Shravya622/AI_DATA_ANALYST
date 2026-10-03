"""
Custom exception classes for the AI-powered Data Analyst application.

All domain-specific errors are defined here so that other modules can raise
and catch typed exceptions rather than bare ``Exception`` instances.
"""


class SessionNotFoundError(Exception):
    """Raised when a requested session ID does not exist in the store."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class DatasetNotFoundError(Exception):
    """Raised when a requested dataset filename is not found in a session."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class UnknownToolError(Exception):
    """Raised when the tool registry receives an unregistered tool name."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigError(Exception):
    """Raised when application configuration is invalid or incomplete."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
