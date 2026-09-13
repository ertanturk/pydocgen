"""Central configuration settings and constants for pydocgen."""

from __future__ import annotations

from typing import Final

# Service & Keyring Credentials
SERVICE_NAME: Final[str] = "pydocgen"
ACCOUNT_NAME: Final[str] = "gemini-api-key"
ENV_API_KEY_NAMES: Final[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
MIN_KEY_LENGTH: Final[int] = 20
INVALID_KEY_CONTROL_CHARS: Final[tuple[str, ...]] = ("\n", "\r", "\t", "\0")
KEY_MASK_PREFIX_LENGTH: Final[int] = 4
KEY_MASK_SUFFIX_LENGTH: Final[int] = 4

# Analysis & Source Parsing
PYTHON_FILE_EXTENSION: Final[str] = ".py"
DEFAULT_FILE_ENCODING: Final[str] = "utf-8-sig"
DEFAULT_UNKNOWN_FILENAME: Final[str] = "<unknown>"
FUNCTION_ID_HASH_LENGTH: Final[int] = 12


# Batching & Token Estimation
# 1 token ≈ 3.5 characters for code constructs (operators, indentation, identifiers)
CHARS_PER_TOKEN: Final[float] = 3.5
# Fixed estimate for system prompt, schema constraints, and format instructions
DEFAULT_PROMPT_OVERHEAD_TOKENS: Final[int] = 500
# Structural JSON framing per function (keys, quotes, function_id wrapper)
PER_FUNCTION_OVERHEAD_TOKENS: Final[int] = 40
DEFAULT_MAX_BATCH_TOKENS: Final[int] = 12000
DEFAULT_MAX_FUNCTIONS_PER_BATCH: Final[int] = 25
BATCH_ID_HASH_LENGTH: Final[int] = 8


# Concurrency & Execution Pool
DEFAULT_RPM: Final[int] = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[float] = 60.0
DEFAULT_MAX_CONCURRENCY: Final[int] = 3
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_INITIAL_BACKOFF: Final[float] = 1.0
DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0


# Telemetry & Logging
DEFAULT_LOGGER_ROOT: Final[str] = "pydocgen"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT_CONSOLE: Final[str] = "console"
LOG_FORMAT_JSON: Final[str] = "json"
DEFAULT_LOG_FORMAT: Final[str] = LOG_FORMAT_CONSOLE
DEFAULT_CONSOLE_LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_REDACTED_MASK: Final[str] = "[REDACTED]"
RESERVED_LOG_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "message",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)
SENSITIVE_KEY_NAMES: Final[frozenset[str]] = frozenset(
    {"key", "api_key", "apikey", "secret", "password", "token", "auth_token"}
)

__all__ = [
    "ACCOUNT_NAME",
    "BATCH_ID_HASH_LENGTH",
    "CHARS_PER_TOKEN",
    "DEFAULT_BACKOFF_MULTIPLIER",
    "DEFAULT_CONSOLE_LOG_FORMAT",
    "DEFAULT_FILE_ENCODING",
    "DEFAULT_INITIAL_BACKOFF",
    "DEFAULT_LOG_DATE_FORMAT",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOGGER_ROOT",
    "DEFAULT_MAX_BATCH_TOKENS",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_FUNCTIONS_PER_BATCH",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_PROMPT_OVERHEAD_TOKENS",
    "DEFAULT_RATE_LIMIT_WINDOW_SECONDS",
    "DEFAULT_REDACTED_MASK",
    "DEFAULT_RPM",
    "DEFAULT_UNKNOWN_FILENAME",
    "ENV_API_KEY_NAMES",
    "FUNCTION_ID_HASH_LENGTH",
    "INVALID_KEY_CONTROL_CHARS",
    "KEY_MASK_PREFIX_LENGTH",
    "KEY_MASK_SUFFIX_LENGTH",
    "LOG_FORMAT_CONSOLE",
    "LOG_FORMAT_JSON",
    "MIN_KEY_LENGTH",
    "PER_FUNCTION_OVERHEAD_TOKENS",
    "PYTHON_FILE_EXTENSION",
    "RESERVED_LOG_RECORD_KEYS",
    "SENSITIVE_KEY_NAMES",
    "SERVICE_NAME",
]
