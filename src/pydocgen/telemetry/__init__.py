"""Telemetry and structured logging subsystem for pydocgen."""

from pydocgen.telemetry.config import (
    configure_telemetry,
    get_logger,
    reset_telemetry,
)
from pydocgen.telemetry.context import (
    bind_context,
    clear_context,
    get_context,
    set_context,
    update_context,
)
from pydocgen.telemetry.filters import (
    SensitiveDataFilter,
    redact_sensitive_data,
    redact_sensitive_string,
)
from pydocgen.telemetry.formatters import (
    ConsoleFormatter,
    JsonFormatter,
)
from pydocgen.telemetry.logger import TelemetryLogger
from pydocgen.telemetry.timing import timer

__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
    "SensitiveDataFilter",
    "TelemetryLogger",
    "bind_context",
    "clear_context",
    "configure_telemetry",
    "get_context",
    "get_logger",
    "redact_sensitive_data",
    "redact_sensitive_string",
    "reset_telemetry",
    "set_context",
    "timer",
    "update_context",
]
