"""Rate limiting module for controlling API invocation frequencies."""

from __future__ import annotations

import asyncio
import time
from collections import deque

from pydocgen.config.settings import (
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_RPM,
)
from pydocgen.errors.exceptions import ConcurrencyError
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Sliding-window rate limiter enforcing a requests-per-minute (RPM) ceiling.

    Attributes:
        rpm: Maximum allowed requests in rolling window. A value of 0 disables limiting.
        window_seconds: Rolling window size in seconds (defaults to 60.0).
    """

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_RPM,
        window_seconds: float = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        if (
            not isinstance(requests_per_minute, int)
            or isinstance(requests_per_minute, bool)
            or requests_per_minute < 0
        ):
            raise ConcurrencyError("requests_per_minute must be a non-negative integer.")

        if (
            not isinstance(window_seconds, (int, float))
            or isinstance(window_seconds, bool)
            or window_seconds <= 0
        ):
            raise ConcurrencyError("window_seconds must be a positive number.")

        self.rpm = requests_per_minute
        self._window_seconds = float(window_seconds)
        self._timestamps: deque[float] = deque()
        self._lock: asyncio.Lock | None = None

    @property
    def window_seconds(self) -> float:
        """Rolling window duration in seconds."""
        return self._window_seconds

    @property
    def active_requests(self) -> int:
        """Number of recorded requests within the current rolling window."""
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= self._window_seconds:
            self._timestamps.popleft()
        return len(self._timestamps)

    def reset(self) -> None:
        """Reset the limiter state by clearing all recorded timestamps."""
        self._timestamps.clear()

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        """Wait until a request slot is available in the current rolling window."""
        if self.rpm <= 0:
            return

        async with self._get_lock():
            while True:
                now = time.monotonic()

                # Evict timestamps outside the rolling window
                while self._timestamps and now - self._timestamps[0] >= self._window_seconds:
                    self._timestamps.popleft()

                # If quota is available, record timestamp and return
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(time.monotonic())
                    logger.debug(
                        "Rate limiter slot acquired",
                        extra={
                            "rpm": self.rpm,
                            "window_seconds": self._window_seconds,
                            "active_requests": len(self._timestamps),
                        },
                    )
                    break

                # Quota exhausted: sleep until oldest timestamp leaves window
                sleep_duration = self._window_seconds - (now - self._timestamps[0])
                if sleep_duration > 0:
                    logger.warning(
                        "Rate limit ceiling reached; throttling request",
                        extra={
                            "rpm": self.rpm,
                            "sleep_duration": sleep_duration,
                            "active_requests": len(self._timestamps),
                        },
                    )
                    await asyncio.sleep(sleep_duration)


__all__ = ["RateLimiter"]
