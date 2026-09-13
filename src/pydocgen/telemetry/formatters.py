"""Formatters for structured JSON and readable console telemetry."""

import datetime
import json
import logging
import traceback
from typing import Any

from pydocgen.config.settings import (
    DEFAULT_CONSOLE_LOG_FORMAT,
    DEFAULT_LOG_DATE_FORMAT,
    RESERVED_LOG_RECORD_KEYS,
)


def _default_json_serializer(obj: Any) -> str:
    """Fallback serializer for non-serializable objects."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return str(obj)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for structured telemetry."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(record.created, tz=datetime.UTC).isoformat()

        log_data: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Extract extra contextual fields
        context: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in RESERVED_LOG_RECORD_KEYS and not key.startswith("_"):
                context[key] = value

        if context:
            log_data["context"] = context

        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "stack_trace": traceback.format_exception(*record.exc_info),
            }
        elif record.exc_text:
            log_data["exception"] = {"message": record.exc_text}

        return json.dumps(log_data, default=_default_json_serializer)


class ConsoleFormatter(logging.Formatter):
    """Formats log records as human-readable lines with contextual key-value pairs."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(
            fmt=fmt or DEFAULT_CONSOLE_LOG_FORMAT,
            datefmt=datefmt or DEFAULT_LOG_DATE_FORMAT,
        )

    def format(self, record: logging.LogRecord) -> str:
        base_message = super().format(record)

        # Collect extra context items
        extra_parts: list[str] = []
        for key, value in record.__dict__.items():
            if key not in RESERVED_LOG_RECORD_KEYS and not key.startswith("_"):
                extra_parts.append(f"{key}={value!r}")

        if extra_parts:
            return f"{base_message} | {' '.join(extra_parts)}"
        return base_message


__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
]
