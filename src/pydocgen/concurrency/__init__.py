"""Concurrency package providing rate limiting and bounded worker execution pools."""

from __future__ import annotations

from pydocgen.concurrency.executor import BatchTaskResult, run_batch_pool
from pydocgen.concurrency.limiter import RateLimiter
from pydocgen.errors.exceptions import ConcurrencyError

__all__ = [
    "BatchTaskResult",
    "ConcurrencyError",
    "RateLimiter",
    "run_batch_pool",
]
