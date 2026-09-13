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


class FileValidationError(PyDocGenError):
    """Raised when a target file path fails validation (missing, wrong type, outside root)."""


class ParseError(PyDocGenError):
    """Raised when a Python file cannot be parsed into an AST."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        super().__init__(message)
        self.line = line
        self.column = column
