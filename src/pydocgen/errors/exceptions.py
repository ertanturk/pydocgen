"""Custom exceptions for the pydocgen package."""


class PyDocGenError(Exception):
    """Base exception for all pydocgen errors."""


class CredentialError(PyDocGenError):
    """Base exception for credential and authentication errors."""


class InvalidCredentialError(CredentialError, ValueError):
    """Raised when an API key or credential is invalid, empty, or malformed."""


class CredentialNotFoundError(CredentialError):
    """Raised when a requested credential is not found in secure storage."""


class CredentialAlreadyExistsError(CredentialError):
    """Raised when attempting to save a credential that already exists in storage."""


class CredentialStorageError(CredentialError):
    """Raised when an error occurs while interacting with secure credential storage."""


class ConfigurationError(PyDocGenError):
    """Raised when application configuration is invalid or missing."""


class ProviderError(PyDocGenError):
    """Base exception for LLM provider errors (e.g., Gemini)."""


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication with an LLM provider fails."""


class AnalysisError(PyDocGenError):
    """Raised when AST parsing or source code analysis fails."""


class GenerationError(PyDocGenError):
    """Raised when docstring generation fails."""
