"""Logging filters for redacting sensitive secrets and API keys."""

import logging
import re
from typing import Any

from pydocgen.config.settings import (
    DEFAULT_REDACTED_MASK,
    KEY_MASK_PREFIX_LENGTH,
    KEY_MASK_SUFFIX_LENGTH,
    SENSITIVE_KEY_NAMES,
)

# Pattern for Google/Gemini API keys (e.g. AIzaSy...)
_API_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z_-]{16,50}\b")
_BEARER_PATTERN = re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)


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
        return f"{s[:KEY_MASK_PREFIX_LENGTH]}...{s[-KEY_MASK_SUFFIX_LENGTH:]}"

    text = _API_KEY_PATTERN.sub(mask_key, text)
    text = _BEARER_PATTERN.sub(rf"Bearer {DEFAULT_REDACTED_MASK}", text)
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
            if any(name in str(k).lower() for name in SENSITIVE_KEY_NAMES):
                if isinstance(v, str) and v.startswith("AIza"):
                    result[k] = redact_sensitive_string(v)
                else:
                    result[k] = DEFAULT_REDACTED_MASK
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
