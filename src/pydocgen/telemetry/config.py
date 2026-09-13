"""Configuration and initialization for the pydocgen telemetry subsystem."""

import logging
import sys
from pathlib import Path
from typing import TextIO

from pydocgen.config.settings import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOGGER_ROOT,
    LOG_FORMAT_CONSOLE,
    LOG_FORMAT_JSON,
)
from pydocgen.telemetry.filters import SensitiveDataFilter
from pydocgen.telemetry.formatters import ConsoleFormatter, JsonFormatter
from pydocgen.telemetry.logger import TelemetryLogger


def _resolve_level(level: str | int) -> int:
    """Resolve level name or integer into standard logging level integer."""
    if isinstance(level, str):
        level_upper = level.upper()
        resolved = getattr(logging, level_upper, None)
        if isinstance(resolved, int):
            return resolved
        raise ValueError(f"Invalid logging level: '{level}'")
    if isinstance(level, int):
        return level
    raise TypeError(f"Logging level must be string or integer, got: {type(level).__name__}")


def get_logger(name: str | None = None) -> TelemetryLogger:
    """Retrieve a context-aware TelemetryLogger instance.

    Args:
        name: Name of the logger, typically __name__. If None, uses root 'pydocgen'.

    Returns:
        TelemetryLogger wrapping the resolved standard logger.
    """
    if not name:
        logger_name = DEFAULT_LOGGER_ROOT
    elif name == DEFAULT_LOGGER_ROOT or name.startswith(f"{DEFAULT_LOGGER_ROOT}."):
        logger_name = name
    else:
        logger_name = f"{DEFAULT_LOGGER_ROOT}.{name}"

    raw_logger = logging.getLogger(logger_name)
    return TelemetryLogger(raw_logger)


def configure_telemetry(
    level: str | int = DEFAULT_LOG_LEVEL,
    format_type: str = DEFAULT_LOG_FORMAT,
    stream: TextIO | None = None,
    log_file: Path | str | None = None,
    propagate: bool = False,
) -> TelemetryLogger:
    """Configure telemetry handlers, formatters, and filters for pydocgen.

    Args:
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        format_type: Formatter type, either 'console' or 'json'.
        stream: Output text stream for console handler (defaults to sys.stderr).
        log_file: Optional path to write log output to a file.
        propagate: Whether to propagate log records to parent root logger.

    Returns:
        Root TelemetryLogger for 'pydocgen'.
    """
    numeric_level = _resolve_level(level)
    root_logger = logging.getLogger(DEFAULT_LOGGER_ROOT)
    root_logger.setLevel(numeric_level)
    root_logger.propagate = propagate

    # Clear existing handlers to prevent duplicate output upon re-configuration
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    # Determine formatter
    format_lower = format_type.lower()
    if format_lower == LOG_FORMAT_JSON:
        formatter: logging.Formatter = JsonFormatter()
    elif format_lower == LOG_FORMAT_CONSOLE:
        formatter = ConsoleFormatter()
    else:
        raise ValueError(
            f"Unsupported format_type: '{format_type}'. Expected '{LOG_FORMAT_CONSOLE}' or '{LOG_FORMAT_JSON}'."
        )

    # Attach StreamHandler
    out_stream = stream if stream is not None else sys.stderr
    stream_handler = logging.StreamHandler(out_stream)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(SensitiveDataFilter())
    root_logger.addHandler(stream_handler)

    # Attach optional FileHandler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(file_handler)

    return TelemetryLogger(root_logger)


def reset_telemetry() -> None:
    """Reset telemetry logger handlers and state to default unconfigured state."""
    root_logger = logging.getLogger(DEFAULT_LOGGER_ROOT)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.NOTSET)
    root_logger.propagate = True
    root_logger.addHandler(logging.NullHandler())


# Ensure default NullHandler is present so unconfigured library imports are silent
_pydocgen_root = logging.getLogger(DEFAULT_LOGGER_ROOT)
if not _pydocgen_root.handlers:
    _pydocgen_root.addHandler(logging.NullHandler())


__all__ = [
    "configure_telemetry",
    "get_logger",
    "reset_telemetry",
]
