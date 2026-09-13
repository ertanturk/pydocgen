"""Exceptions and error handling for pydocgen."""

from pydocgen.errors.exceptions import (
    ConfigurationError,
    CredentialAlreadyExistsError,
    CredentialError,
    CredentialNotFoundError,
    CredentialStorageError,
    FileValidationError,
    InvalidCredentialError,
    ParseError,
    PyDocGenError,
)

__all__ = [
    "ConfigurationError",
    "CredentialAlreadyExistsError",
    "CredentialError",
    "CredentialNotFoundError",
    "CredentialStorageError",
    "FileValidationError",
    "InvalidCredentialError",
    "ParseError",
    "PyDocGenError",
]
