"""Timing context manager and measurement utilities for operations."""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def timer(
    logger: logging.Logger | logging.LoggerAdapter,
    operation_name: str,
    level: int = logging.INFO,
    **extra: Any,
) -> Generator[dict[str, Any]]:
    """Context manager measuring execution duration and logging completion or failure.

    Args:
        logger: Target logger or adapter.
        operation_name: Human-readable name of the operation.
        level: Log level for start and success messages (default: INFO).
        **extra: Additional metadata fields to include in log records.

    Yields:
        Dictionary for capturing runtime metrics to be logged upon completion.
    """
    start_time = time.perf_counter()
    metrics: dict[str, Any] = {}

    logger.log(
        level,
        f"Starting {operation_name}",
        extra={"operation": operation_name, "status": "started", **extra},
    )

    try:
        yield metrics
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            f"Failed {operation_name} after {duration_ms}ms: {exc}",
            extra={
                "operation": operation_name,
                "status": "failed",
                "duration_ms": duration_ms,
                "error": str(exc),
                **extra,
                **metrics,
            },
            exc_info=True,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.log(
            level,
            f"Completed {operation_name} in {duration_ms}ms",
            extra={
                "operation": operation_name,
                "status": "completed",
                "duration_ms": duration_ms,
                **extra,
                **metrics,
            },
        )


__all__ = ["timer"]
