"""Formatters for structured JSON and readable console telemetry."""

import datetime
import json
import logging
import traceback
from typing import Any

# Standard LogRecord attributes that should not be dumped into custom context
_LOGRECORD_RESERVED_ATTRS = frozenset(
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
            if key not in _LOGRECORD_RESERVED_ATTRS and not key.startswith("_"):
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
            fmt=fmt or "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base_message = super().format(record)

        # Collect extra context items
        extra_parts: list[str] = []
        for key, value in record.__dict__.items():
            if key not in _LOGRECORD_RESERVED_ATTRS and not key.startswith("_"):
                extra_parts.append(f"{key}={value!r}")

        if extra_parts:
            return f"{base_message} | {' '.join(extra_parts)}"
        return base_message


__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
]
