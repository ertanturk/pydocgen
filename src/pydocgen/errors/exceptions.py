"""Custom exceptions for the pydocgen package."""

from enum import IntEnum


class ExitCode(IntEnum):
    """Standard system exit codes adhering to BSD/POSIX sysexits.h."""

    OK = 0  # Successful termination
    USAGE = 64  # Command line usage error
    DATAERR = 65  # Data format error (e.g., malformed LLM response or syntax error)
    NOINPUT = 66  # Cannot open input file
    UNAVAILABLE = 69  # Service unavailable / third-party API offline
    SOFTWARE = 70  # Internal software error
    TEMPFAIL = 75  # Temporary failure; indicates the caller can retry
    CONFIG = 78  # Configuration error (e.g., missing or invalid API key)


class PyDocGenError(Exception):
    """Base exception for all pydocgen errors."""

    exit_code: ExitCode = ExitCode.SOFTWARE

    def __init__(self, message: str = "", exit_code: ExitCode | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class PydocgenError(PyDocGenError):
    """Base exception for all pydocgen failures."""


class CredentialError(PydocgenError):
    """Base exception for credential and authentication errors."""

    exit_code: ExitCode = ExitCode.CONFIG


class InvalidCredentialError(CredentialError, ValueError):
    """Raised when an API key or credential is invalid, empty, or malformed."""

    exit_code: ExitCode = ExitCode.USAGE


class CredentialNotFoundError(CredentialError):
    """Raised when a requested credential is not found in secure storage."""


class CredentialAlreadyExistsError(CredentialError):
    """Raised when attempting to save a credential that already exists in storage."""


class CredentialStorageError(CredentialError):
    """Raised when an error occurs while interacting with secure credential storage."""


class ConfigurationError(PydocgenError):
    """Raised when application configuration is invalid or missing."""

    exit_code: ExitCode = ExitCode.CONFIG


class FileValidationError(PydocgenError):
    """Raised when a target file path fails validation (missing, wrong type, outside root)."""

    exit_code: ExitCode = ExitCode.NOINPUT


class ParseError(PydocgenError):
    """Raised when a Python file cannot be parsed into an AST."""

    exit_code: ExitCode = ExitCode.DATAERR

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        exit_code: ExitCode | None = None,
    ) -> None:
        super().__init__(message, exit_code=exit_code)
        self.line = line
        self.column = column


class BatchError(PydocgenError):
    """Raised when batch formation fails or constraints are violated."""

    exit_code: ExitCode = ExitCode.DATAERR


class ConcurrencyError(PydocgenError):
    """Raised when execution parameters are invalid or execution fails completely."""

    exit_code: ExitCode = ExitCode.SOFTWARE


class ProviderError(PydocgenError):
    """Base exception for provider communication failures."""

    exit_code: ExitCode = ExitCode.UNAVAILABLE


class AuthenticationError(ProviderError):
    """Raised when authentication credentials are missing or rejected."""

    exit_code: ExitCode = ExitCode.CONFIG


class RateLimitError(ProviderError):
    """Raised when the provider rejects a request due to RPM/TPM quota exhaustion."""

    exit_code: ExitCode = ExitCode.TEMPFAIL


class ProviderTimeoutError(ProviderError):
    """Raised when a request to the provider times out."""

    exit_code: ExitCode = ExitCode.TEMPFAIL


class ProviderResponseError(ProviderError):
    """Raised when the provider returns unparseable or schema-invalid data."""

    exit_code: ExitCode = ExitCode.DATAERR


class SourceEditError(PydocgenError):
    """Raised when calculating, previewing, or applying source code edits fails."""

    exit_code: ExitCode = ExitCode.SOFTWARE


class PipelineExecutionError(PydocgenError):
    """Raised when an unrecoverable failure aborts the pipeline run."""

    exit_code: ExitCode = ExitCode.SOFTWARE


class FormattingError(PydocgenError, ValueError):
    """Raised when formatting generated documentation into a docstring fails."""

    exit_code: ExitCode = ExitCode.DATAERR


class ValidationError(PydocgenError, ValueError):
    """Raised when validating generated documentation fails."""

    exit_code: ExitCode = ExitCode.DATAERR


class GenerationError(PydocgenError):
    """Raised when documentation generation fails."""

    exit_code: ExitCode = ExitCode.SOFTWARE
