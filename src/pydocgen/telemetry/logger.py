"""Context-aware telemetry logger adapter supporting metadata binding and timers."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pydocgen.config.settings import RESERVED_LOG_RECORD_KEYS
from pydocgen.telemetry.context import get_context
from pydocgen.telemetry.timing import timer


class TelemetryLogger(logging.LoggerAdapter):
    """Context-aware logger adapter supporting structured extra data and timing."""

    def __init__(
        self,
        logger: logging.Logger,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(logger, extra or {})

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        """Merge global contextvars, adapter extra, and invocation-specific extra.

        Also protects against overwriting standard LogRecord attributes.
        """
        merged_extra: dict[str, Any] = {**get_context(), **(self.extra or {})}
        if "extra" in kwargs and isinstance(kwargs["extra"], dict):
            merged_extra.update(kwargs["extra"])

        safe_extra: dict[str, Any] = {}
        for k, v in merged_extra.items():
            if k in RESERVED_LOG_RECORD_KEYS:
                safe_extra[f"extra_{k}"] = v
            else:
                safe_extra[k] = v

        kwargs["extra"] = safe_extra
        return msg, kwargs

    def bind(self, **kwargs: Any) -> TelemetryLogger:
        """Return a child TelemetryLogger with additional bound context."""
        new_extra = {**(self.extra or {}), **kwargs}
        return TelemetryLogger(self.logger, new_extra)

    @contextmanager
    def timed(
        self, operation_name: str, level: int = logging.INFO, **extra: Any
    ) -> Generator[dict[str, Any]]:
        """Context manager measuring execution duration and logging completion or failure.

        Args:
            operation_name: Name of the operation being executed.
            level: Logging level for start and completion messages.
            **extra: Additional metadata fields to attach.

        Yields:
            Dictionary for capturing runtime metrics.
        """
        with timer(self, operation_name, level=level, **extra) as metrics:
            yield metrics


__all__ = ["TelemetryLogger"]
