"""Logging filters for redacting sensitive secrets and API keys."""

import logging
import re
from typing import Any

# Pattern for Google/Gemini API keys (e.g. AIzaSy...)
_API_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z_-]{16,50}\b")
_BEARER_PATTERN = re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)
_SENSITIVE_KEY_NAMES = frozenset(
    {"key", "api_key", "apikey", "secret", "password", "token", "auth_token"}
)


def redact_sensitive_string(text: str) -> str:
    """Redact known sensitive tokens and API keys from a string.

    Args:
        text: Input string that might contain secrets.

    Returns:
        Sanitized string with secrets masked.
    """
    if not isinstance(text, str):
        return text

    def mask_key(match: re.Match[str]) -> str:
        s = match.group(0)
        return f"{s[:4]}...{s[-4:]}"

    text = _API_KEY_PATTERN.sub(mask_key, text)
    text = _BEARER_PATTERN.sub(r"Bearer [REDACTED]", text)
    return text


def redact_sensitive_data(obj: Any) -> Any:
    """Recursively redact sensitive strings within dictionaries, lists, or primitives.

    Args:
        obj: Arbitrary data structure.

    Returns:
        Sanitized copy of data structure.
    """
    if isinstance(obj, str):
        return redact_sensitive_string(obj)
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if any(name in str(k).lower() for name in _SENSITIVE_KEY_NAMES):
                if isinstance(v, str) and v.startswith("AIza"):
                    result[k] = redact_sensitive_string(v)
                else:
                    result[k] = "[REDACTED]"
            else:
                result[k] = redact_sensitive_data(v)
        return result
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_sensitive_data(x) for x in obj)
    return obj


class SensitiveDataFilter(logging.Filter):
    """Filter that sanitizes log record messages and arguments to prevent leaking secrets."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_string(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_sensitive_data(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_sensitive_data(x) for x in record.args)

        return True


__all__ = [
    "SensitiveDataFilter",
    "redact_sensitive_data",
    "redact_sensitive_string",
]
